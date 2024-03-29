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
Green='\033[0;32m'
Orange='\033[0;33m'
Cyan='\033[0;36m'
Purple='\033[0;35m'
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
        echo -e "${Red}No simulation data detected.${NC}"
        return 1
    fi

    echo -e "${Cyan}Detected data-set(s):${NC}"
    for first_part in "${!unique_first_parts[@]}"; do
        local interpreted_name=$(interpret_folder_name "$first_part")
        echo -e "${Purple}- $interpreted_name ($first_part)${NC}"
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
        echo -e "${Green}Data seem OK. You can run integrity check to be sure.${NC}"
    else
        echo -e "${Orange}Incomplete simulation data. Consider cleaning up and re-generating the data.${NC}"
    fi
}

check_data_integrity() {
    local name=$1
    echo
    echo -e "Checking data integrity in project ${Orange}$name${NC}, this may take a while..."
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
        echo -e "${Green}All data integrity checks have passed.${NC}"
    else
        echo
        echo -e "${Orange}Some data integrity checks failed.${NC}"
    fi
}

main_menu() {
    local name=$1
    # Main loop
    while true; do
        echo -e "${Cyan}"
        echo "Please select a task:"
        echo
        echo "(D)ata Generation"
        echo "(M)odel Training"
        echo "(V)alidation"
        echo "(O)ptimization"
        echo
        echo "(C)heck integrity of existing data"
        echo "(R)emove existing data"
        echo -e "${Red}(Q)uit"
        echo -e "${NC}"

        read -p "Enter your choice: " task
        case $task in
            D)
                while true; do
                    echo -e "${Cyan}"
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
                    echo -e "${Red}(Q)uit"
                    echo -e "${NC}"

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
                                        echo -e "${Red}Invalid input. ${NC}Please enter a combination of B, D, P, E, or single A, N, Q."
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
                                    if [ -e "../Config/bd_sim.yaml" ]; then
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
                                    if [ -e "../Config/ddd_sim.yaml" ]; then
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
                                    if [ -e "../Config/pbd_sim.yaml" ]; then
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
                                    if [ -e "../Config/eve_sim.yaml" ]; then
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
                    echo -e "${Cyan}"
                    echo "Please select one GNN model to train:"
                    echo
                    echo "(1) for Simple GCN"
                    echo "(2) for GCN+DiffPool"
                    echo "(3) for Graph Transformer"
                    echo
                    echo "(N) to go back"
                    echo -e "${Red}(Q)uit"
                    echo -e "${NC}"

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
                                echo -e "${Red}No data-set found.${NC}"
                                continue
                            else
                                echo -e "${Cyan}"
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
                                    echo -e "${NC}"
                                    echo "Training model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            if [ -e "../Config/bd_train_gnn.yaml" ]; then
                                                echo -e "${Blue}Training model on Birth-Death Trees...${NC}"
                                                # Logic for Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_bd_pars_est_model_training.sh "$name" "BD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_bd_pars_est_model_training.sh "$name" "BD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death simple GNN model."
                                            fi
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            if [ -e "../Config/ddd_train_gnn.yaml" ]; then
                                                echo -e "${Blue}Training model on Diversity-Dependent-Diversification Trees...${NC}"
                                                # Logic for Diversity-Dependent-Diversification Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_ddd_pars_est_model_training.sh "$name" "DDD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_ddd_pars_est_model_training.sh "$name" "DDD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification simple GNN model."
                                            fi
                                            ;;
                                        "Protracted Birth-Death")
                                            if [ -e "../Config/pbd_train_gnn.yaml" ]; then
                                                echo -e "${Blue}Training model on Protracted Birth-Death Trees...${NC}"
                                                # Logic for Protracted Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_pbd_pars_est_model_training.sh "$name" "PBD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_pbd_pars_est_model_training.sh "$name" "PBD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death simple GNN model."
                                            fi
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            if [ -e "../Config/eve_train_gnn.yaml" ]; then
                                                echo -e "${Blue}Training model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                                # Logic for Evolutionary-Relatedness-Dependent Trees
                                                echo "Submitting jobs for TES"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_model_training.sh "$name" "EVE_FREE_TES" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_model_training.sh "$name" "EVE_FREE_TES" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_model_training.sh "$name" "EVE_FREE_TES" "nnd"
                                                echo "Submitting job for TAS"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_model_training.sh "$name" "EVE_FREE_TAS" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_model_training.sh "$name" "EVE_FREE_TAS" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_model_training.sh "$name" "EVE_FREE_TAS" "nnd"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent simple GNN model."
                                            fi
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
                                echo -e "${Red}No data-set found.${NC}"
                                continue
                            else
                                echo -e "${Cyan}"
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
                                    echo -e "${NC}"
                                    echo "Training model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            if [ -e "../Config/bd_train_diffpool.yaml" ]; then
                                                echo -e "${Blue}Training model on Birth-Death Trees...${NC}"
                                                # Logic for Birth-Death Trees
                                                local max_gnn_depth
                                                max_gnn_depth=$(grep 'max_gnn_depth:' ../Config/bd_train_diffpool.yaml | awk '{print $2}')
                                                echo "Max GNN depth: $max_gnn_depth"
                                                for (( i=1; i<=max_gnn_depth; i++ )); do
                                                    echo "Submitting job for TES with GNN depth $i"
                                                    sbatch submit_bd_pars_est_model_training_diffpool.sh "$name" "BD_FREE_TES" "$i"
                                                    sbatch submit_bd_pars_est_model_training_diffpool_full.sh "$name" "BD_FREE_TES" "$i"
                                                    echo "Submitting job for TAS with GNN depth $i"
                                                    sbatch submit_bd_pars_est_model_training_diffpool.sh "$name" "BD_FREE_TAS" "$i"
                                                    sbatch submit_bd_pars_est_model_training_diffpool_full.sh "$name" "BD_FREE_TAS" "$i"
                                                done
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death DiffPool model."
                                            fi
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            if [ -e "../Config/ddd_train_diffpool.yaml" ]; then
                                                echo -e "${Blue}Training model on Diversity-Dependent-Diversification Trees...${NC}"
                                                # Logic for Diversity-Dependent-Diversification Trees
                                                local max_gnn_depth
                                                max_gnn_depth=$(grep 'max_gnn_depth:' ../Config/ddd_train_diffpool.yaml | awk '{print $2}')
                                                echo "Max GNN depth: $max_gnn_depth"
                                                for (( i=1; i<=max_gnn_depth; i++ )); do
                                                    echo "Submitting job for TES with GNN depth $i"
                                                    sbatch submit_ddd_pars_est_model_training_diffpool.sh "$name" "DDD_FREE_TES" "$i"
                                                    #sbatch submit_ddd_pars_est_model_training_diffpool_full.sh "$name" "DDD_FREE_TES" "$i"
                                                    #echo "Submitting job for TAS with GNN depth $i"
                                                    #sbatch submit_ddd_pars_est_model_training_diffpool.sh "$name" "DDD_FREE_TAS" "$i"
                                                    #sbatch submit_ddd_pars_est_model_training_diffpool_full.sh "$name" "DDD_FREE_TAS" "$i"
                                                done
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification DiffPool model."
                                            fi
                                            ;;
                                        "Protracted Birth-Death")
                                            if [ -e "../Config/pbd_train_diffpool.yaml" ]; then
                                                echo -e "${Blue}Training model on Protracted Birth-Death Trees...${NC}"
                                                # Logic for Protracted Birth-Death Trees
                                                local max_gnn_depth
                                                max_gnn_depth=$(grep 'max_gnn_depth:' ../Config/pbd_train_diffpool.yaml | awk '{print $2}')
                                                echo "Max GNN depth: $max_gnn_depth"
                                                for (( i=1; i<=max_gnn_depth; i++ )); do
                                                    echo "Submitting job for TES with GNN depth $i"
                                                    sbatch submit_pbd_pars_est_model_training_diffpool.sh "$name" "PBD_FREE_TES" "$i"
                                                    sbatch submit_pbd_pars_est_model_training_diffpool_full.sh "$name" "PBD_FREE_TES" "$i"
                                                    echo "Submitting job for TAS with GNN depth $i"
                                                    sbatch submit_pbd_pars_est_model_training_diffpool.sh "$name" "PBD_FREE_TAS" "$i"
                                                    sbatch submit_pbd_pars_est_model_training_diffpool_full.sh "$name" "PBD_FREE_TAS" "$i"
                                                done
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death DiffPool model."
                                            fi
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            if [ -e "../Config/eve_train_diffpool.yaml" ]; then
                                                echo -e "${Blue}Training model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                                # Logic for Evolutionary-Relatedness-Dependent Trees
                                                local max_gnn_depth
                                                max_gnn_depth=$(grep 'max_gnn_depth:' ../Config/eve_train_diffpool.yaml | awk '{print $2}')
                                                echo "Max GNN depth: $max_gnn_depth"
                                                for (( i=1; i<=max_gnn_depth; i++ )); do
                                                    echo "Submitting jobs for TES with GNN depth $i"
                                                    echo "Submitting PD"
                                                    sbatch submit_eve_pars_est_model_training_diffpool.sh "$name" "EVE_FREE_TES" "pd" "$i"
                                                    echo "Submitting ED"
                                                    sbatch submit_eve_pars_est_model_training_diffpool.sh "$name" "EVE_FREE_TES" "ed" "$i"
                                                    echo "Submitting NND"
                                                    sbatch submit_eve_pars_est_model_training_diffpool.sh "$name" "EVE_FREE_TES" "nnd" "$i"
                                                    echo "Submitting job for TAS with GNN depth $i"
                                                    echo "Submitting PD"
                                                    sbatch submit_eve_pars_est_model_training_diffpool.sh "$name" "EVE_FREE_TAS" "pd" "$i"
                                                    echo "Submitting ED"
                                                    sbatch submit_eve_pars_est_model_training_diffpool.sh "$name" "EVE_FREE_TAS" "ed" "$i"
                                                    echo "Submitting NND"
                                                    sbatch submit_eve_pars_est_model_training_diffpool.sh "$name" "EVE_FREE_TAS" "nnd" "$i"
                                                done
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent DiffPool model."
                                            fi
                                            ;;
                                    esac
                                done
                            fi
                            ;;
                        3)
                            echo -e "${Cyan}"
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
                                echo -e "${Red}No data-set found.${NC}"
                                continue
                            else
                                echo -e "${Cyan}"
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
                                    echo -e "${NC}"
                                    echo "Training model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            if [ -e "../Config/bd_train_gps.yaml" ]; then
                                                echo -e "${Blue}Training model on Birth-Death Trees...${NC}"
                                                # Logic for Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_bd_pars_est_model_training_gps.sh "$name" "BD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_bd_pars_est_model_training_gps.sh "$name" "BD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death Graph Transformer model."
                                            fi
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            if [ -e "../Config/ddd_train_gps.yaml" ]; then
                                                echo -e "${Blue}Training model on Diversity-Dependent-Diversification Trees...${NC}"
                                                # Logic for Diversity-Dependent-Diversification Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_ddd_pars_est_model_training_gps.sh "$name" "DDD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_ddd_pars_est_model_training_gps.sh "$name" "DDD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification Graph Transformer model."
                                            fi
                                            ;;
                                        "Protracted Birth-Death")
                                            if [ -e "../Config/pbd_train_gps.yaml" ]; then
                                                echo -e "${Blue}Training model on Protracted Birth-Death Trees...${NC}"
                                                # Logic for Protracted Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_pbd_pars_est_model_training_gps.sh "$name" "PBD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_pbd_pars_est_model_training_gps.sh "$name" "PBD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death Graph Transformer model."
                                            fi
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            if [ -e "../Config/eve_train_gps.yaml" ]; then
                                                echo -e "${Blue}Training model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                                # Logic for Evolutionary-Relatedness-Dependent Trees
                                                echo "Submitting job for TES"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_model_training_gps.sh "$name" "EVE_FREE_TES" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_model_training_gps.sh "$name" "EVE_FREE_TES" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_model_training_gps.sh "$name" "EVE_FREE_TES" "nnd"
                                                echo "Submitting job for TAS"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_model_training_gps.sh "$name" "EVE_FREE_TAS" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_model_training_gps.sh "$name" "EVE_FREE_TAS" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_model_training_gps.sh "$name" "EVE_FREE_TAS" "nnd"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent Graph Transformer model."
                                            fi
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
            O)
                while true; do
                    echo -e "${Cyan}"
                    echo "Please select one GNN model to optimize:"
                    echo
                    echo "(1) for Simple GCN"
                    echo "(2) for GCN+DiffPool"
                    echo "(3) for Graph Transformer"
                    echo
                    echo "(N) to go back"
                    echo -e "${Red}(Q)uit"
                    echo -e "${NC}"

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
                                echo -e "${Red}No data-set found.${NC}"
                                continue
                            else
                                echo -e "${Cyan}"
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
                                    echo -e "${NC}"
                                    echo "Optimizing model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            if [ -e "../Config/bd_opt_gnn.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Birth-Death Trees...${NC}"
                                                # Logic for Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_bd_pars_est_opt.sh "$name" "BD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_bd_pars_est_opt.sh "$name" "BD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death simple GNN model."
                                            fi
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            if [ -e "../Config/ddd_opt_gnn.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Diversity-Dependent-Diversification Trees...${NC}"
                                                # Logic for Diversity-Dependent-Diversification Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_ddd_pars_est_opt.sh "$name" "DDD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_ddd_pars_est_opt.sh "$name" "DDD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification simple GNN model."
                                            fi
                                            ;;
                                        "Protracted Birth-Death")
                                            if [ -e "../Config/pbd_opt_gnn.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Protracted Birth-Death Trees...${NC}"
                                                # Logic for Protracted Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_pbd_pars_est_opt.sh "$name" "PBD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_pbd_pars_est_opt.sh "$name" "PBD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death simple GNN model."
                                            fi
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            if [ -e "../Config/eve_opt_gnn.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                                # Logic for Evolutionary-Relatedness-Dependent Trees
                                                echo "Submitting jobs for TES"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_opt.sh "$name" "EVE_FREE_TES" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_opt.sh "$name" "EVE_FREE_TES" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_opt.sh "$name" "EVE_FREE_TES" "nnd"
                                                echo "Submitting job for TAS"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_opt.sh "$name" "EVE_FREE_TAS" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_opt.sh "$name" "EVE_FREE_TAS" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_opt.sh "$name" "EVE_FREE_TAS" "nnd"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent simple GNN model."
                                            fi
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
                                echo -e "${Red}No data-set found.${NC}"
                                continue
                            else
                                echo -e "${Cyan}"
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
                                    echo -e "${NC}"
                                    echo "Optimizing model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            if [ -e "../Config/bd_opt_diffpool.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Birth-Death Trees...${NC}"
                                                # Logic for Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_bd_pars_est_opt_diffpool.sh "$name" "BD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_bd_pars_est_opt_diffpool.sh "$name" "BD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death DiffPool model."
                                            fi
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            if [ -e "../Config/ddd_opt_diffpool.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Diversity-Dependent-Diversification Trees...${NC}"
                                                # Logic for Diversity-Dependent-Diversification Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_ddd_pars_est_opt_diffpool.sh "$name" "DDD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_ddd_pars_est_opt_diffpool.sh "$name" "DDD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification DiffPool model."
                                            fi
                                            ;;
                                        "Protracted Birth-Death")
                                            if [ -e "../Config/pbd_opt_diffpool.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Protracted Birth-Death Trees...${NC}"
                                                # Logic for Protracted Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_pbd_pars_est_opt_diffpool.sh "$name" "PBD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_pbd_pars_est_opt_diffpool.sh "$name" "PBD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death DiffPool model."
                                            fi
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            if [ -e "../Config/eve_opt_diffpool.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                                # Logic for Evolutionary-Relatedness-Dependent Trees
                                                echo "Submitting job for TES"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_opt_diffpool_reg.sh "$name" "EVE_FREE_TES" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_opt_diffpool_reg.sh "$name" "EVE_FREE_TES" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_opt_diffpool_reg.sh "$name" "EVE_FREE_TES" "nnd"
                                                echo "Submitting job for TAS"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_opt_diffpool_reg.sh "$name" "EVE_FREE_TAS" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_opt_diffpool_reg.sh "$name" "EVE_FREE_TAS" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_opt_diffpool_reg.sh "$name" "EVE_FREE_TAS" "nnd"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent DiffPool model."
                                            fi
                                            ;;
                                    esac
                                done
                            fi
                            ;;
                        3)
                            echo -e "${Cyan}"
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
                                echo -e "${Red}No data-set found.${NC}"
                                continue
                            else
                                echo -e "${Cyan}"
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
                                    echo -e "${NC}"
                                    echo "Optimizing model on selected data-set: $folder_type"
                                    # Logic based on selected data-set type
                                    case $folder_type in
                                        "Birth-Death")
                                            if [ -e "../Config/bd_opt_gps.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Birth-Death Trees...${NC}"
                                                # Logic for Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_bd_pars_est_opt_gps.sh "$name" "BD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_bd_pars_est_opt_gps.sh "$name" "BD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death Graph Transformer model."
                                            fi
                                            ;;
                                        "Diversity-Dependent-Diversification")
                                            if [ -e "../Config/ddd_opt_gps.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Diversity-Dependent-Diversification Trees...${NC}"
                                                # Logic for Diversity-Dependent-Diversification Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_ddd_pars_est_opt_gps.sh "$name" "DDD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_ddd_pars_est_opt_gps.sh "$name" "DDD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification Graph Transformer model."
                                            fi
                                            ;;
                                        "Protracted Birth-Death")
                                            if [ -e "../Config/pbd_opt_gps.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Protracted Birth-Death Trees...${NC}"
                                                # Logic for Protracted Birth-Death Trees
                                                echo "Submitting job for TES"
                                                sbatch submit_pbd_pars_est_opt_gps.sh "$name" "PBD_FREE_TES"
                                                echo "Submitting job for TAS"
                                                sbatch submit_pbd_pars_est_opt_gps.sh "$name" "PBD_FREE_TAS"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death Graph Transformer model."
                                            fi
                                            ;;
                                        "Evolutionary-Relatedness-Dependent")
                                            if [ -e "../Config/eve_opt_gps.yaml" ]; then
                                                echo -e "${Blue}Optimizing model on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                                # Logic for Evolutionary-Relatedness-Dependent Trees
                                                echo "Submitting job for TES"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_opt_gps.sh "$name" "EVE_FREE_TES" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_opt_gps.sh "$name" "EVE_FREE_TES" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_opt_gps.sh "$name" "EVE_FREE_TES" "nnd"
                                                echo "Submitting job for TAS"
                                                echo "Submitting PD"
                                                sbatch submit_eve_pars_est_opt_gps.sh "$name" "EVE_FREE_TAS" "pd"
                                                echo "Submitting ED"
                                                sbatch submit_eve_pars_est_opt_gps.sh "$name" "EVE_FREE_TAS" "ed"
                                                echo "Submitting NND"
                                                sbatch submit_eve_pars_est_opt_gps.sh "$name" "EVE_FREE_TAS" "nnd"
                                            else
                                                echo
                                                echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent Graph Transformer model."
                                            fi
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
                    echo -e "${Red}No data-set found.${NC}"
                else
                    echo -e "${Cyan}"
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
                        echo -e "${NC}"
                        echo "No selection made."
                    else
                        if [ "$task" == "V" ]; then
                            for folder_type in "${selected_folder_types[@]}"; do
                                echo -e "${NC}"
                                echo "Performing validation on selected data-set: $folder_type"
                                # Validation logic here based on folder type
                                case $folder_type in
                                    "Birth-Death")
                                        if [ -e "../Config/bd_val_gnn.yaml" ]; then
                                          echo -e "${Blue}Performing simple GNN validation on Birth-Death Trees...${NC}"
                                          sbatch submit_bd_pars_est_val.sh "$name" "BD_VAL_TES"
                                          sbatch submit_bd_pars_est_val.sh "$name" "BD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death simple GNN model."
                                        fi
                                        if [ -e "../Config/bd_val_diffpool.yaml" ]; then
                                          echo -e "${Blue}Performing DiffPool validation on Birth-Death Trees...${NC}"
                                          sbatch submit_bd_pars_est_val_diffpool.sh "$name" "BD_VAL_TES"
                                          sbatch submit_bd_pars_est_val_diffpool.sh "$name" "BD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death DiffPool model."
                                        fi
                                        if [ -e "../Config/bd_val_gps.yaml" ]; then
                                          echo -e "${Blue}Performing Graph Transformer validation on Birth-Death Trees...${NC}"
                                          sbatch submit_bd_pars_est_val_gps.sh "$name" "BD_VAL_TES"
                                          sbatch submit_bd_pars_est_val_gps.sh "$name" "BD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Birth-Death Graph Transformer model."
                                        fi
                                        ;;
                                    "Diversity-Dependent-Diversification")
                                        if [ -e "../Config/ddd_val_gnn.yaml" ]; then
                                          echo -e "${Blue}Performing simple GNN validation on Diversity-Dependent-Diversification Trees...${NC}"
                                          sbatch submit_ddd_pars_est_val.sh "$name" "DDD_VAL_TES"
                                          sbatch submit_ddd_pars_est_val.sh "$name" "DDD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification simple GNN model."
                                        fi
                                        if [ -e "../Config/ddd_val_diffpool.yaml" ]; then
                                          echo -e "${Blue}Performing DiffPool validation on Diversity-Dependent-Diversification Trees...${NC}"
                                          sbatch submit_ddd_pars_est_val_diffpool.sh "$name" "DDD_VAL_TES"
                                          sbatch submit_ddd_pars_est_val_diffpool.sh "$name" "DDD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification DiffPool model."
                                        fi
                                        if [ -e "../Config/ddd_val_gps.yaml" ]; then
                                          echo -e "${Blue}Performing Graph Transformer validation on Diversity-Dependent-Diversification Trees...${NC}"
                                          sbatch submit_ddd_pars_est_val_gps.sh "$name" "DDD_VAL_TES"
                                          sbatch submit_ddd_pars_est_val_gps.sh "$name" "DDD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Diversity-Dependent-Diversification Graph Transformer model."
                                        fi
                                        ;;
                                    "Protracted Birth-Death")
                                        if [ -e "../Config/pbd_val_gnn.yaml" ]; then
                                          echo -e "${Blue}Performing simple GNN validation on Protracted Birth-Death Trees...${NC}"
                                          sbatch submit_pbd_pars_est_val.sh "$name" "PBD_VAL_TES"
                                          sbatch submit_pbd_pars_est_val.sh "$name" "PBD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death simple GNN model."
                                        fi
                                        if [ -e "../Config/pbd_val_diffpool.yaml" ]; then
                                          echo -e "${Blue}Performing DiffPool validation on Protracted Birth-Death Trees...${NC}"
                                          sbatch submit_pbd_pars_est_val_diffpool.sh "$name" "PBD_VAL_TES"
                                          sbatch submit_pbd_pars_est_val_diffpool.sh "$name" "PBD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death DiffPool model."
                                        fi
                                        if [ -e "../Config/pbd_val_gps.yaml" ]; then
                                          echo -e "${Blue}Performing Graph Transformer validation on Protracted Birth-Death Trees...${NC}"
                                          sbatch submit_pbd_pars_est_val_gps.sh "$name" "PBD_VAL_TES"
                                          sbatch submit_pbd_pars_est_val_gps.sh "$name" "PBD_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Protracted Birth-Death Graph Transformer model."
                                        fi
                                        ;;
                                    "Evolutionary-Relatedness-Dependent")
                                        if [ -e "../Config/eve_val_gnn.yaml" ]; then
                                          echo -e "${Blue}Performing simple GNN validation on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                          sbatch submit_eve_pars_est_val.sh "$name" "EVE_VAL_TES"
                                          sbatch submit_eve_pars_est_val.sh "$name" "EVE_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent simple GNN model."
                                        fi
                                        if [ -e "../Config/eve_val_diffpool.yaml" ]; then
                                          echo -e "${Blue}Performing DiffPool validation on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                          sbatch submit_eve_pars_est_val_diffpool.sh "$name" "EVE_VAL_TES"
                                          sbatch submit_eve_pars_est_val_diffpool.sh "$name" "EVE_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent DiffPool model."
                                        fi
                                        if [ -e "../Config/eve_val_gps.yaml" ]; then
                                          echo -e "${Blue}Performing Graph Transformer validation on Evolutionary-Relatedness-Dependent Trees...${NC}"
                                          sbatch submit_eve_pars_est_val_gps.sh "$name" "EVE_VAL_TES"
                                          sbatch submit_eve_pars_est_val_gps.sh "$name" "EVE_VAL_TAS"
                                        else
                                          echo
                                          echo -e "${Red}ERROR: ${NC}Missing configuration file for Evolutionary-Relatedness-Dependent Graph Transformer model."
                                        fi
                                        ;;
                                esac
                            done
                        else
                            echo -e "${NC}"
                            echo "Selected data-set for removal:"
                            printf '%s\n' "${selected_folder_types[@]}"
                            echo
                            read -p $'\033[0;31mAre you sure you want to remove this data-set? (y/N): \033[0m' confirm
                            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                                echo "This may take a while..."
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
echo -e "${Cyan}"
echo "  █████╗ ██████╗  █████╗ ███╗   ███╗ "
echo " ██╔══██╗██╔══██╗██╔══██╗████╗ ████║ "
echo " ███████║██║  ██║███████║██╔████╔██║ "
echo " ██╔══██║██║  ██║██╔══██║██║╚██╔╝██║ "
echo " ██║  ██║██████╔╝██║  ██║██║ ╚═╝ ██║ "
echo " ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝ "

echo
echo
echo "Welcome to the Automated DAta Manager for eveGNN."
echo "Author: Tianjian Qin"
echo "Version: 1.0 (20231130)"

echo
echo -e "Current project folder: ${Orange}$name${NC}"
echo

verify_data_at_start "$name"

main_menu "$name"

