#!/bin/bash

# ADAM.sh: Automated Data Manager for eveGNN
# Check if a path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_project>"
    exit 1
fi

name=$1

# Check if the specified folder exists
if [ ! -d "$name" ]; then
    echo "The specified project folder '$name' does not exist."
    exit 1
fi

# Define some colors
Red='\033[0;31m'
Blue='\033[0;34m'
Orange='\033[0;33m'
NC='\033[0m' # No Color

# Function list
interpret_folder_name() {
    local folder_name=$1
    local func_part=${folder_name%%_*}

    case $func_part in
        BD)
            echo "Birth-Death"
            ;;
        DDD)
            echo "Diversity-Dependent-Diversification"
            ;;
        PBD)
            echo "Protracted Birth-Death"
            ;;
        EVE)
            echo "Evolutionary-Relatedness-Dependent"
            ;;
        *)
            echo "Unknown"
            ;;
    esac
}

interpret_combination() {
    local combination=$1
    local A=${combination%%_*}
    local B=${combination##*_}

    case $A in
        FREE)
            A="training"
            ;;
        VAL)
            A="out-of-sample validation"
            ;;
    esac

    echo "$A data ($B trees)"
}

verify_data_at_start() {
    local name=$1
    echo "Detecting simulation data..."

    # List directories matching the pattern and identify unique first parts
    local raw_folders=("$name"/*_*_*)
    declare -A unique_first_parts
    for folder in "${raw_folders[@]}"; do
        if [ -d "$folder" ]; then
            local first_part=$(basename "$folder" | cut -d '_' -f1)
            unique_first_parts[$first_part]=1
        fi
    done

    if [ ${#unique_first_parts[@]} -eq 0 ]; then
        echo "No simulation data detected."
        return 1
    fi

    echo "Detected data-set(s):"
    for first_part in "${!unique_first_parts[@]}"; do
        local interpreted_name=$(interpret_folder_name "$first_part")
        echo "- $interpreted_name ($first_part)"
    done

    # Check for required combinations
    local failed_check=0
    for first_part in "${!unique_first_parts[@]}"; do
        local part2s=("FREE" "VAL")
        local part3s=("TES" "TAS")

        for part2 in "${part2s[@]}"; do
            for part3 in "${part3s[@]}"; do
                local combination="${first_part}_${part2}_${part3}"
                if [ ! -d "$name/$combination" ]; then
                    local interpreted_combination=$(interpret_combination "${part2}_${part3}")
                    echo -e "${Orange}WARNING: ${NC}Missing $interpreted_combination in the $(interpret_folder_name "$first_part") data-set."
                    failed_check=1
                fi
            done
        done

        # Check for MLE_TES combination
        if  [ "$first_part" != "EVE" ]; then
            if [ ! -d "$name/${first_part}_MLE_TES" ]; then
                echo -e "${Orange}WARNING: ${NC}Missing Maximum Likelihood Estimation (MLE) results in the $(interpret_folder_name "$first_part") data-set."
                failed_check=1
            fi
        fi
    done

    if [ $failed_check -eq 0 ]; then
        echo "Data seem OK. You can run integrity check to be sure."
    else
        echo "Incomplete simulation data. Consider cleaning up and re-generating the data."
    fi
}

check_data_integrity() {
    local name=$1
    echo
    echo "Checking data integrity in project $name, this may take a while..."
    echo

    # Possible combinations excluding MLE_TES
    local combinations=("FREE_TES" "FREE_TAS" "VAL_TES" "VAL_TAS")
    local all_checks_passed=1  # Flag to track if all checks have passed

    for combination in "${combinations[@]}"; do
        for folder in "$name"/*_"$combination"; do
            if [ -d "$folder" ]; then
                # Extract and interpret parts
                local part1=$(basename "$folder" | cut -d '_' -f1)
                local part2=$(basename "$folder" | cut -d '_' -f2)
                local part3=$(basename "$folder" | cut -d '_' -f3)
                local interpreted_part1=$(interpret_folder_name "$part1")
                local interpreted_part2=$(interpret_combination "${part2}_${part3}")
                local found_format=0

                # Check for GNN and GPS folders
                if [ -d "$folder/GNN" ]; then
                    found_format=1
                    echo "Found GNN data format in $interpreted_part1 $interpreted_part2"
                    local count_tree=$(find "$folder/GNN/tree" -maxdepth 1 -name "*.rds" | wc -l)
                    local count_el=$(find "$folder/GNN/tree/EL" -maxdepth 1 -name "*.rds" | wc -l)
                    echo "Count in tree: $count_tree, Count in tree/EL: $count_el"
                    if [ "$count_tree" -ne "$count_el" ]; then
                        echo -e "${Orange}WARNING: ${NC}Inconsistent file counts in GNN format."
                        all_checks_passed=0
                    fi
                fi
                if [ -d "$folder/GPS" ]; then
                    found_format=1
                    echo "Found GPS data format in $interpreted_part1 $interpreted_part2"
                    local count_tree=$(find "$folder/GPS/tree" -maxdepth 1 -name "*.rds" | wc -l)
                    local count_edge=$(find "$folder/GPS/tree/edge" -maxdepth 1 -name "*.rds" | wc -l)
                    local count_node=$(find "$folder/GPS/tree/node" -maxdepth 1 -name "*.rds" | wc -l)
                    echo "Count in tree: $count_tree, Count in tree/edge: $count_edge, Count in tree/node: $count_node"
                    if [ "$count_tree" -ne "$count_edge" ] || [ "$count_tree" -ne "$count_node" ]; then
                        echo -e "${Orange}WARNING: ${NC}Inconsistent file counts in GPS format."
                        all_checks_passed=0
                    fi
                fi

                # Check if no format was found
                if [ $found_format -eq 0 ]; then
                    echo
                    echo -e "${Orange}WARNING: ${NC}Neither GNN nor GPS data formats found in $interpreted_part1 $interpreted_part2."
                    all_checks_passed=0
                fi
            fi
        done
    done

    # Final report
    if [ $all_checks_passed -eq 1 ]; then
        echo
        echo "All data integrity checks have passed."
    else
        echo
        echo "Some data integrity checks failed."
    fi
}

main_menu() {
    local name=$1
    # Main loop
    while true; do
        echo
        echo "Please select a task:"
        echo
        echo "(D)ata Generation"
        echo "(M)odel Training"
        echo "(V)alidation"
        echo
        echo "(C)heck integrity of existing data"
        echo "(R)emove existing data"
        echo "(Q)uit"
        echo

        read -p "Enter your choice: " task
        case $task in
            D)
                while true; do
                    echo
                    echo "Please select one or more data-set(s) that should be generated, must be a combination of B, D, P, E or single A, N, Q."
                    echo
                    echo "(B)irth-Death Trees"
                    echo "(D)iversity-Dependent-Diversification Trees"
                    echo "(P)rotracted Birth-Death Trees"
                    echo "(E)volutionary-Relatedness-Dependent Trees"
                    echo
                    echo "(A)ll the above"
                    echo
                    echo "(N) to go back"
                    echo "(Q)uit"
                    echo

                    read -p "Enter your choice: " sim_func_input
                    valid_input=true
                    selected_scenarios=()

                    # Check if the input is 'A', 'N', or 'Q', which are handled separately
                    case $sim_func_input in
                        A)
                            selected_scenarios=("B" "D" "P" "E")
                            ;;
                        N)
                            break
                            ;;
                        Q)
                            exit 0
                            ;;
                        *)
                            # Loop through each character in the input
                            for (( i=0; i<${#sim_func_input}; i++ )); do
                                sim_func=${sim_func_input:$i:1}
                                case $sim_func in
                                    B|D|P|E)
                                        selected_scenarios+=("$sim_func")
                                        ;;
                                    *)
                                        echo
                                        echo "Invalid input. Please enter a combination of B, D, P, E, or single A, N, Q."
                                        echo
                                        valid_input=false
                                        break
                                        ;;
                                esac
                            done
                            ;;
                    esac

                    if [ "$valid_input" = true ] ; then
                        echo
                        echo "Selected scenarios: ${selected_scenarios[*]}"
                        # Loop through selected_scenarios array to handle each one
                        for scenario in "${selected_scenarios[@]}"; do
                            case $scenario in
                                B)
                                    if [ -d "../Config/bd_sim.yaml" ]; then
                                        echo
                                        echo -e "${Blue}Submitting Birth-Death simulation job...${NC}}"
                                        # Add logic for Birth-Death Trees here
                                        sbatch submit_bd_pars_est_free_data.sh "$name"
                                    else
                                        echo
                                        echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death simulation."
                                    fi
                                    ;;
                                D)
                                    if [ -d "../Config/ddd_sim.yaml" ]; then
                                        echo
                                        echo -e "${Blue}Submitting Diversity-Dependent-Diversification simulation job...${NC}"
                                        # Add logic for Diversity-Dependent-Diversification Trees here
                                        sbatch submit_ddd_pars_est_free_data.sh "$name"
                                    else
                                        echo
                                        echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification simulation."
                                    fi
                                    ;;
                                P)
                                    if [ -d "../Config/pbd_sim.yaml" ]; then
                                        echo
                                        echo -e "${Blue}Submitting Protracted Birth-Death simulation job...${NC}"
                                        # Add logic for Protracted Birth-Death Trees here
                                        sbatch submit_pbd_pars_est_free_data.sh "$name"
                                    else
                                        echo
                                        echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death simulation."
                                    fi
                                    ;;
                                E)
                                    if [ -d "../Config/eve_sim.yaml" ]; then
                                        echo
                                        echo -e "${Blue}Submitting Evolutionary-Relatedness-Dependent simulation job...${NC}"
                                        # Add logic for Evolutionary-Relatedness-Dependent Trees here
                                        sbatch submit_eve_pars_est_free_data.sh "$name"
                                    else
                                        echo
                                        echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent simulation."
                                    fi
                                    ;;
                            esac
                        done
                        break
                    fi
                done
                ;;
            M)
                while true; do
                    echo
                    echo "Please select one GNN model to train:"
                    echo
                    echo "(1) for Simple GCN"
                    echo "(2) for GCN+DiffPool"
                    echo "(3) for Graph Transformer"
                    echo
                    echo "(N) to go back"
                    echo "(Q)uit"
                    echo

                    read -p "Enter your choice: " model_choice
                    case $model_choice in
                        1)
                            echo
                            echo "Selected model: $model_choice"
                            # List unique folder types using shell's glob pattern
                            local -A folder_types
                            local unique_folder_types
                            folder_types=()
                            unique_folder_types=()

                            for folder in "$name"/*_*_*; do
                                if [ -d "$folder" ]; then
                                    function_name=$(interpret_folder_name "$(basename "$folder")")
                                    if [ "$function_name" != "Unknown" ] && [ -z "${folder_types[$function_name]}" ]; then
                                        folder_types[$function_name]=1
                                        unique_folder_types+=("$function_name")
                                    fi
                                fi
                            done

                            if [ ${#unique_folder_types[@]} -eq 0 ]; then
                                echo
                                echo "No data-set found."
                                continue
                            else
                                echo
                                echo "Found the following data-set type(s):"
                                echo
                                echo "Select a data-set type or 'All' to proceed with all data-sets:"
                                select folder_type_option in "${unique_folder_types[@]}" "All" "Back"; do
                                    case $folder_type_option in
                                        "All")
                                            selected_folder_types=("${unique_folder_types[@]}")
                                            break
                                            ;;
                                        "Back")
                                            break 2
                                            ;;
                                        *)
                                            selected_folder_types=("$folder_type_option")
                                            break
                                            ;;
                                    esac
                                done

                                for folder_type in "${selected_folder_types[@]}"; do
                                    echo
                                    echo "Training model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            echo -e "${Blue}Training model on Birth-Death Trees...${NC}"
                                            # Logic for Birth-Death Trees
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            echo -e "${Blue}Training model on Diversity-Dependent-Diversification Trees...${NC}"
                                            # Logic for Diversity-Dependent-Diversification Trees
                                            ;;
                                        "Protracted Birth-Death")
                                            echo -e "${Blue}Training model on Protracted Birth-Death Trees...${NC}"
                                            # Logic for Protracted Birth-Death Trees
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            echo -e "${Blue}Training model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                            # Logic for Evolutionary-Relatedness-Dependent Trees
                                            ;;
                                    esac
                                done
                            fi
                            ;;
                        2)
                            echo
                            echo "Selected model: $model_choice"
                            # List unique folder types using shell's glob pattern
                            local -A folder_types
                            local unique_folder_types
                            folder_types=()
                            unique_folder_types=()

                            for folder in "$name"/*_*_*; do
                                if [ -d "$folder" ]; then
                                    function_name=$(interpret_folder_name "$(basename "$folder")")
                                    if [ "$function_name" != "Unknown" ] && [ -z "${folder_types[$function_name]}" ]; then
                                        folder_types[$function_name]=1
                                        unique_folder_types+=("$function_name")
                                    fi
                                fi
                            done

                            if [ ${#unique_folder_types[@]} -eq 0 ]; then
                                echo
                                echo "No data-set found."
                                continue
                            else
                                echo
                                echo "Found the following data-set type(s):"
                                echo
                                echo "Select a data-set or 'All' to proceed with all data-sets:"
                                select folder_type_option in "${unique_folder_types[@]}" "All" "Back"; do
                                    case $folder_type_option in
                                        "All")
                                            selected_folder_types=("${unique_folder_types[@]}")
                                            break
                                            ;;
                                        "Back")
                                            break 2
                                            ;;
                                        *)
                                            selected_folder_types=("$folder_type_option")
                                            break
                                            ;;
                                    esac
                                done

                                for folder_type in "${selected_folder_types[@]}"; do
                                    echo
                                    echo "Training model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            echo -e "${Blue}Training model on Birth-Death Trees...${NC}"
                                            # Logic for Birth-Death Trees
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            echo -e "${Blue}Training model on Diversity-Dependent-Diversification Trees...${NC}"
                                            # Logic for Diversity-Dependent-Diversification Trees
                                            ;;
                                        "Protracted Birth-Death")
                                            echo -e "${Blue}Training model on Protracted Birth-Death Trees...${NC}"
                                            # Logic for Protracted Birth-Death Trees
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            echo -e "${Blue}Training model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                            # Logic for Evolutionary-Relatedness-Dependent Trees
                                            ;;
                                    esac
                                done
                            fi
                            ;;
                        3)
                            echo
                            echo "Selected GNN model: $model_choice"
                            # List unique folder types using shell's glob pattern
                            local -A folder_types
                            local unique_folder_types
                            folder_types=()
                            unique_folder_types=()

                            for folder in "$name"/*_*_*; do
                                if [ -d "$folder" ]; then
                                    function_name=$(interpret_folder_name "$(basename "$folder")")
                                    if [ "$function_name" != "Unknown" ] && [ -z "${folder_types[$function_name]}" ]; then
                                        folder_types[$function_name]=1
                                        unique_folder_types+=("$function_name")
                                    fi
                                fi
                            done

                            if [ ${#unique_folder_types[@]} -eq 0 ]; then
                                echo
                                echo "No data-set found."
                                continue
                            else
                                echo
                                echo "Found the following data-set type(s):"
                                echo
                                echo "Select a data-set type or 'All' to proceed with all data-sets:"
                                select folder_type_option in "${unique_folder_types[@]}" "All" "Back"; do
                                    case $folder_type_option in
                                        "All")
                                            selected_folder_types=("${unique_folder_types[@]}")
                                            break
                                            ;;
                                        "Back")
                                            break 2
                                            ;;
                                        *)
                                            selected_folder_types=("$folder_type_option")
                                            break
                                            ;;
                                    esac
                                done

                                for folder_type in "${selected_folder_types[@]}"; do
                                    echo
                                    echo "Training model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            echo -e "${Blue}Training model on Birth-Death Trees...${NC}"
                                            # Logic for Birth-Death Trees
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            echo -e "${Blue}Training model on Diversity-Dependent-Diversification Trees...${NC}"
                                            # Logic for Diversity-Dependent-Diversification Trees
                                            ;;
                                        "Protracted Birth-Death")
                                            echo -e "${Blue}Training model on Protracted Birth-Death Trees...${NC}"
                                            # Logic for Protracted Birth-Death Trees
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            echo -e "${Blue}Training model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                            # Logic for Evolutionary-Relatedness-Dependent Trees
                                            ;;
                                    esac
                                done
                            fi
                            ;;
                        N)
                            break
                            ;;
                        Q)
                            exit 0
                            ;;
                        *)
                            echo
                            echo "Aborting..."
                            exit 0
                            ;;
                    esac
                done
                ;;
            V|R)
                local -A folder_types
                local unique_folder_types
                local selected_folder_types
                folder_types=()
                unique_folder_types=()
                selected_folder_types=()

                for folder in "$name"/*_*_*; do
                    if [ -d "$folder" ]; then
                        function_name=$(interpret_folder_name "$(basename "$folder")")
                        if [ "$function_name" != "Unknown" ] && [ -z "${folder_types[$function_name]}" ]; then
                            folder_types[$function_name]=1
                            unique_folder_types+=("$function_name")
                        fi
                    fi
                done

                if [ ${#unique_folder_types[@]} -eq 0 ]; then
                    echo
                    echo "No data-set found."
                else
                    echo
                    echo "Found the following data-set(s):"
                    echo
                    echo "Select a data-set or 'All' to proceed with all data-sets:"
                    select folder_type_option in "${unique_folder_types[@]}" "All" "Cancel"; do
                        case $folder_type_option in
                            "All")
                                selected_folder_types=("${unique_folder_types[@]}")
                                break
                                ;;
                            "Cancel")
                                break
                                ;;
                            *)
                                selected_folder_types=("$folder_type_option")
                                break
                                ;;
                        esac
                    done

                    if [ ${#selected_folder_types[@]} -eq 0 ]; then
                        echo
                        echo "No selection made."
                    else
                        if [ "$task" == "V" ]; then
                            for folder_type in "${selected_folder_types[@]}"; do
                                echo
                                echo "Performing validation on selected data-set: $folder_type"
                                # Validation logic here based on folder type
                                case $folder_type in
                                    "Birth-Death")
                                        echo
                                        echo -e "${Blue}Performing validation on Birth-Death Trees...${NC}"
                                        # Logic for Birth-Death Trees
                                        ;;
                                    "Diversity-Dependent-Diversification")
                                        echo
                                        echo -e "${Blue}Performing validation on Diversity-Dependent-Diversification Trees...${NC}"
                                        # Logic for Diversity-Dependent-Diversification Trees
                                        ;;
                                    "Protracted Birth-Death")
                                        echo
                                        echo -e "${Blue}Performing validation on Protracted Birth-Death Trees...${NC}"
                                        # Logic for Protracted Birth-Death Trees
                                        ;;
                                    "Evolutionary-Relatedness-Dependent")
                                        echo
                                        echo -e "${Blue}Performing validation on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                        # Logic for Evolutionary-Relatedness-Dependent Trees
                                        ;;
                                esac
                            done
                        else
                            echo
                            echo "Selected data-set for removal:"
                            printf '%s\n' "${selected_folder_types[@]}"
                            echo
                            read -p "${Red}Are you sure you want to remove this data-set? (y/N): ${NC}" confirm
                            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                                for folder_type in "${selected_folder_types[@]}"; do
                                    for folder in "$name"/*_*_*; do
                                        if [ -d "$folder" ] && [[ "$(interpret_folder_name "$(basename "$folder")")" == "$folder_type" ]]; then
                                            echo
                                            echo -e "${Blue}Removing $folder...${NC}"
                                            rm -rf "$folder"
                                        fi
                                    done
                                done
                            else
                                echo
                                echo -e "${Blue}Removal cancelled.${NC}"
                            fi
                        fi
                    fi
                fi
                ;;
            C)
                check_data_integrity "$name"
                ;;
            Q)
                echo
                echo "Aborting..."
                exit 0
                ;;
            *)
                echo
                echo "Invalid choice. Aborting..."
                exit 0
                ;;
        esac
    done
}

###############################################
# Start Page: Automated Data Manager for eveGNN
echo
echo
echo
echo "  █████╗ ██████╗  █████╗ ███╗   ███╗ "
echo " ██╔══██╗██╔══██╗██╔══██╗████╗ ████║ "
echo " ███████║██║  ██║███████║██╔████╔██║ "
echo " ██╔══██║██║  ██║██╔══██║██║╚██╔╝██║ "
echo " ██║  ██║██████╔╝██║  ██║██║ ╚═╝ ██║ "
echo " ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝ "

echo
echo
echo "Welcome to the Automated DAta Manager V1 for eveGNN."

echo
echo "Current project folder: $name"
echo

verify_data_at_start "$name"

main_menu "$name"

