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
                    echo "WARNING: Missing $interpreted_combination in the $(interpret_folder_name "$first_part") data-set."
                    failed_check=1
                fi
            done
        done

        # Check for MLE_TES combination
        if [ ! -d "$name/${first_part}_MLE_TES" ]; then
            echo "WARNING: Missing Maximum Likelihood Estimation (MLE) results in the $(interpret_folder_name "$first_part") data-set."
            failed_check=1
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

    # Possible combinations excluding MLE_TES
    local combinations=("FREE_TES" "FREE_TAS" "VAL_TES" "VAL_TAS")
    local all_checks_passed=1  # Flag to track overall integrity status

    for combination in "${combinations[@]}"; do
        for folder in "$name"/*_"$combination"; do
            if [ -d "$folder" ]; then
                # Extract and interpret parts
                local part1=$(basename "$folder" | cut -d '_' -f1)
                local part2=$(basename "$folder" | cut -d '_' -f2)
                local part3=$(basename "$folder" | cut -d '_' -f3)
                local interpreted_part1=$(interpret_folder_name "$part1")
                local found_format=0

                # Check for GNN and GPS folders
                if [ -d "$folder/GNN" ]; then
                    found_format=1
                    echo
                    echo "Found GNN data format in $interpreted_part1 $part2 $part3"
                    local files_tree=($(find "$folder/GNN/tree" -maxdepth 1 -name "*.rds" -exec basename {} \;))
                    local files_el=($(find "$folder/GNN/tree/EL" -maxdepth 1 -name "*.rds" -exec basename {} \;))
                    echo "Count in tree: ${#files_tree[@]}, Count in tree/EL: ${#files_el[@]}"
                    if [ "${#files_tree[@]}" -eq "${#files_el[@]}" ] && [ "${files_tree[*]}" == "${files_el[*]}" ]; then
                        echo "Consistency check passed for GNN format."
                    else
                        echo "WARNING: Inconsistent file counts or names in GNN format."
                        all_checks_passed=0
                    fi
                fi
                if [ -d "$folder/GPS" ]; then
                    found_format=1
                    echo
                    echo "Found GPS data format in $interpreted_part1 $part2 $part3"
                    local files_tree=($(find "$folder/GPS/tree" -maxdepth 1 -name "*.rds" -exec basename {} \;))
                    local files_edge=($(find "$folder/GPS/tree/edge" -maxdepth 1 -name "*.rds" -exec basename {} \;))
                    local files_node=($(find "$folder/GPS/tree/node" -maxdepth 1 -name "*.rds" -exec basename {} \;))
                    echo "Count in tree: ${#files_tree[@]}, Count in tree/edge: ${#files_edge[@]}, Count in tree/node: ${#files_node[@]}"
                    if [ "${#files_tree[@]}" -eq "${#files_edge[@]}" ] && [ "${#files_tree[@]}" -eq "${#files_node[@]}" ] && [ "${files_tree[*]}" == "${files_edge[*]}" ] && [ "${files_tree[*]}" == "${files_node[*]}" ]; then
                        echo "Consistency check passed for GPS format."
                    else
                        echo "WARNING: Inconsistent file counts or names in GPS format."
                        all_checks_passed=0
                    fi
                fi

                # Check if no format was found
                if [ $found_format -eq 0 ]; then
                    echo
                    echo "WARNING: Neither GNN nor GPS data formats found in $interpreted_part1 $part2 $part3."
                    all_checks_passed=0
                fi
            fi
        done
    done

    if [ $all_checks_passed -eq 1 ]; then
        echo
        echo "All data integrity checks have passed."
    else
        echo
        echo "Some data integrity checks have failed."
    fi
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

# Main loop
while true; do
    echo
    echo "Please select a task:"
    echo
    echo "(D)ata Generation"
    echo "(M)odel Training"
    echo "(V)alidation"
    echo
    echo "(C)eck integrity of existing data"
    echo "(R)emove existing data"
    echo "(Q) to abort"
    echo

    read -p "Enter your choice: " task
    case $task in
        D)
            while true; do
                echo
                echo "Please select one or more data-set(s) that should be generated, must be a combination of B, D, P, E or single A, N, Q."
                echo "(B)irth-Death Trees"
                echo "(D)iversity-Dependent-Diversification Trees"
                echo "(P)rotracted Birth-Death Trees"
                echo "(E)volutionary-Relatedness-Dependent Trees"
                echo
                echo "(A)ll the above"
                echo
                echo "(N) togo back"
                echo "(Q) to abort"

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
                                echo
                                echo "Processing Birth-Death Trees..."
                                # Add logic for Birth-Death Trees here
                                ;;
                            D)
                                echo
                                echo "Processing Diversity-Dependent-Diversification Trees..."
                                # Add logic for Diversity-Dependent-Diversification Trees here
                                ;;
                            P)
                                echo
                                echo "Processing Protracted Birth-Death Trees..."
                                # Add logic for Protracted Birth-Death Trees here
                                ;;
                            E)
                                echo
                                echo "Processing Evolutionary-Relatedness-Dependent Trees..."
                                # Add logic for Evolutionary-Relatedness-Dependent Trees here
                                ;;
                        esac
                    done
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
                echo "(Q) to abort"
                echo

                read -p "Enter your choice: " model_choice
                case $model_choice in
                    1|2|3)
                        echo
                        echo "Selected model: $model_choice"
                        # List unique folder types
                        IFS=$'\n' read -r -d '' -a raw_folders <<< "$(find "$name" -type d -name "*_*_*")"
                        declare -A folder_types
                        unique_folder_types=()

                        for folder in "${raw_folders[@]}"; do
                            function_name=$(interpret_folder_name "$(basename "$folder")")
                            if [ "$function_name" != "Unknown" ] && [ -z "${folder_types[$function_name]}" ]; then
                                folder_types[$function_name]=1
                                unique_folder_types+=("$function_name")
                            fi
                        done

                        if [ ${#unique_folder_types[@]} -eq 0 ]; then
                            echo
                            echo "No data-set found."
                            continue
                        else
                            echo
                            echo "Found the following data-set type(s):"
                            selected_folder_types=()
                            while true; do
                                echo
                                echo "Select data-set type(s) or 'Done' to proceed:"
                                select folder_type_option in "${unique_folder_types[@]}" "Done" "Back" "Quit"; do
                                    case $folder_type_option in
                                        "Done")
                                            break 2
                                            ;;
                                        "Back")
                                            break 3
                                            ;;
                                        "Quit")
                                            exit 0
                                            ;;
                                        *)
                                            selected_folder_types+=("$folder_type_option")
                                            echo
                                            echo "Selected data-set(s): ${selected_folder_types[*]}"
                                            break
                                            ;;
                                    esac
                                done
                            done

                            for folder_type in "${selected_folder_types[@]}"; do
                                echo
                                echo "Training model on selected data-set: $folder_type"
                                # Logic based on selected data-set type
                                case $folder_type in
                                    "Birth-Death")
                                        echo
                                        echo "Training model on Birth-Death Trees..."
                                        # Logic for Birth-Death Trees
                                        ;;
                                    "Diversity-Dependent-Diversification")
                                        echo
                                        echo "Training model on Diversity-Dependent-Diversification Trees..."
                                        # Logic for Diversity-Dependent-Diversification Trees
                                        ;;
                                    "Protracted Birth-Death")
                                        echo
                                        echo "Training model on Protracted Birth-Death Trees..."
                                        # Logic for Protracted Birth-Death Trees
                                        ;;
                                    "Evolutionary-Relatedness-Dependent")
                                        echo
                                        echo "Training model on Evolutionary-Relatedness-Dependent Trees..."
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
            IFS=$'\n' read -r -d '' -a raw_folders <<< "$(find "$name" -type d -name "*_*_*")"
            declare -A folder_types
            unique_folder_types=()

            for folder in "${raw_folders[@]}"; do
                function_name=$(interpret_folder_name "$(basename "$folder")")
                if [ "$function_name" != "Unknown" ] && [ -z "${folder_types[$function_name]}" ]; then
                    folder_types[$function_name]=1
                    unique_folder_types+=("$function_name")
                fi
            done

            if [ ${#unique_folder_types[@]} -eq 0 ]; then
                echo
                echo "No data-set found."
            else
                echo
                echo "Found the following data-set type(s):"
                selected_folder_types=()
                while true; do
                    echo
                    echo "Select data-set type(s) or 'Done' to proceed:"
                    select folder_type_option in "${unique_folder_types[@]}" "Done" "Cancel"; do
                        case $folder_type_option in
                            "Done")
                                break 2
                                ;;
                            "Cancel")
                                break 3
                                ;;
                            *)
                                selected_folder_types+=("$folder_type_option")
                                echo
                                echo "Selected: ${selected_folder_types[*]}"
                                break
                                ;;
                        esac
                    done
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
                                    echo "Performing validation on Birth-Death Trees..."
                                    # Logic for Birth-Death Trees
                                    ;;
                                "Diversity-Dependent-Diversification")
                                    echo
                                    echo "Performing validation on Diversity-Dependent-Diversification Trees..."
                                    # Logic for Diversity-Dependent-Diversification Trees
                                    ;;
                                "Protracted Birth-Death")
                                    echo
                                    echo "Performing validation on Protracted Birth-Death Trees..."
                                    # Logic for Protracted Birth-Death Trees
                                    ;;
                                "Evolutionary-Relatedness-Dependent")
                                    echo
                                    echo "Performing validation on Evolutionary-Relatedness-Dependent Trees..."
                                    # Logic for Evolutionary-Relatedness-Dependent Trees
                                    ;;
                            esac
                        done
                    else
                        echo
                        echo "Selected folder types for removal:"
                        printf '%s\n' "${selected_folder_types[@]}"
                        echo
                        read -p "Are you sure you want to remove all folders of these types? (y/N): " confirm
                        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                            for folder_type in "${selected_folder_types[@]}"; do
                                for folder in "${raw_folders[@]}"; do
                                    if [[ "$(interpret_folder_name "$(basename "$folder")")" == "$folder_type" ]]; then
                                        echo
                                        echo "Removing $folder..."
                                        rm -rf "$folder"
                                    fi
                                done
                            done
                        else
                            echo
                            echo "Removal cancelled."
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
