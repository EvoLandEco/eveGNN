#!/bin/bash

# Check if a path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

name=$1

# Check if the specified folder exists
if [ ! -d "$name" ]; then
    echo "The specified folder '$name' does not exist."
    exit 1
fi

# Function to interpret folder names
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
            A="model training"
            ;;
        VAL)
            A="model out-of-sample validation"
            ;;
    esac

    echo "$A data-set ($B trees)"
}

verify_data_integrity() {
    local name=$1
    echo "Verifying simulation data integrity..."

    if [ ! -d "$name" ]; then
        echo "The specified folder '$name' does not exist."
        return 1
    fi

    IFS=$'\n' read -r -d '' -a raw_folders <<< "$(find "$name" -type d -name "*_*_*")"
    declare -A unique_first_parts
    for folder in "${raw_folders[@]}"; do
        function_name=$(interpret_folder_name "$(basename "$folder")")
        if [ "$function_name" != "Unknown" ]; then
            unique_first_parts[$function_name]=1
        fi
    done

    if [ ${#unique_first_parts[@]} -eq 0 ]; then
        echo "No simulation data detected."
        return 1
    fi

    local failed_check=0
    for first_part in "${!unique_first_parts[@]}"; do
        local combinations=("FREE_TES" "FREE_TAS" "VAL_TES" "VAL_TAS")

        for combination in "${combinations[@]}"; do
            local found_combination=0
            for folder in "${raw_folders[@]}"; do
                if [[ "$(basename "$folder")" == "$first_part"_"$combination" ]]; then
                    found_combination=1
                    break
                fi
            done
            if [ $found_combination -eq 0 ]; then
                local interpreted_combination=$(interpret_combination "$combination")
                echo "WARNING: Missing $interpreted_combination in the $first_part data-set."
                failed_check=1
            fi
        done
    done

    if [ $failed_check -eq 0 ]; then
        echo "Passed."
    else
        echo "Data integrity check failed. Consider re-running the simulation."
    fi
}

# Automated Data Manager for eveGNN
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
echo "Welcome to the Automated Data Manager for eveGNN."

echo
verify_data_integrity "$name"
echo

while true; do
    echo "Please select a task:"
    echo
    echo "(D)ata Generation"
    echo "(M)odel Training"
    echo "(V)alidation"
    echo "(R)emove existing data"
    echo "(Q) to abort"

    read -p "Enter your choice: " task
    case $task in
        D)
            while true; do
                echo "Please select one or more data-set(s) that should be generated, must be a combination of B, D, P, E or single A, N, Q."
                echo "(B)irth-Death Trees"
                echo "(D)iversity-Dependent-Diversification Trees"
                echo "(P)rotracted Birth-Death Trees"
                echo "(E)volutionary-Relatedness-Dependent Trees"
                echo "(A)ll the above"
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
                                    echo "Invalid input. Please enter a combination of B, D, P, E, or single A, N, Q."
                                    valid_input=false
                                    break
                                    ;;
                            esac
                        done
                        ;;
                esac

                if [ "$valid_input" = true ] ; then
                    echo "Selected scenarios: ${selected_scenarios[*]}"
                    # Loop through selected_scenarios array to handle each one
                    for scenario in "${selected_scenarios[@]}"; do
                        case $scenario in
                            B)
                                echo "Processing Birth-Death Trees..."
                                # Add logic for Birth-Death Trees here
                                ;;
                            D)
                                echo "Processing Diversity-Dependent-Diversification Trees..."
                                # Add logic for Diversity-Dependent-Diversification Trees here
                                ;;
                            P)
                                echo "Processing Protracted Birth-Death Trees..."
                                # Add logic for Protracted Birth-Death Trees here
                                ;;
                            E)
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
                echo "Please select one GNN model to train:"
                echo "(1) for Simple GCN"
                echo "(2) for GCN+DiffPool"
                echo "(3) for Graph Transformer"
                echo "(N) to go back"
                echo "(Q) to abort"

                read -p "Enter your choice: " model_choice
                case $model_choice in
                    1|2|3)
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
                            echo "No data-set found."
                            continue
                        else
                            echo "Found the following data-set type(s):"
                            selected_folder_types=()
                            while true; do
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
                                            echo "Selected data-set(s): ${selected_folder_types[*]}"
                                            break
                                            ;;
                                    esac
                                done
                            done

                            for folder_type in "${selected_folder_types[@]}"; do
                                echo "Training model on selected data-set: $folder_type"
                                # Logic based on selected data-set type
                                case $folder_type in
                                    "Birth-Death")
                                        echo "Training model on Birth-Death Trees..."
                                        # Logic for Birth-Death Trees
                                        ;;
                                    "Diversity-Dependent-Diversification")
                                        echo "Training model on Diversity-Dependent-Diversification Trees..."
                                        # Logic for Diversity-Dependent-Diversification Trees
                                        ;;
                                    "Protracted Birth-Death")
                                        echo "Training model on Protracted Birth-Death Trees..."
                                        # Logic for Protracted Birth-Death Trees
                                        ;;
                                    "Evolutionary-Relatedness-Dependent")
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
                echo "No data-set found."
            else
                echo "Found the following data-set type(s):"
                selected_folder_types=()
                while true; do
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
                                echo "Selected: ${selected_folder_types[*]}"
                                break
                                ;;
                        esac
                    done
                done

                if [ ${#selected_folder_types[@]} -eq 0 ]; then
                    echo "No selection made."
                else
                    if [ "$task" == "V" ]; then
                        for folder_type in "${selected_folder_types[@]}"; do
                            echo "Performing validation on selected data-set: $folder_type"
                            # Validation logic here based on folder type
                            case $folder_type in
                                "Birth-Death")
                                    echo "Performing validation on Birth-Death Trees..."
                                    # Logic for Birth-Death Trees
                                    ;;
                                "Diversity-Dependent-Diversification")
                                    echo "Performing validation on Diversity-Dependent-Diversification Trees..."
                                    # Logic for Diversity-Dependent-Diversification Trees
                                    ;;
                                "Protracted Birth-Death")
                                    echo "Performing validation on Protracted Birth-Death Trees..."
                                    # Logic for Protracted Birth-Death Trees
                                    ;;
                                "Evolutionary-Relatedness-Dependent")
                                    echo "Performing validation on Evolutionary-Relatedness-Dependent Trees..."
                                    # Logic for Evolutionary-Relatedness-Dependent Trees
                                    ;;
                            esac
                        done
                    else
                        echo "Selected folder types for removal:"
                        printf '%s\n' "${selected_folder_types[@]}"
                        read -p "Are you sure you want to remove all folders of these types? (y/N): " confirm
                        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                            for folder_type in "${selected_folder_types[@]}"; do
                                for folder in "${raw_folders[@]}"; do
                                    if [[ "$(interpret_folder_name "$(basename "$folder")")" == "$folder_type" ]]; then
                                        echo "Removing $folder..."
                                        rm -rf "$folder"
                                    fi
                                done
                            done
                        else
                            echo "Removal cancelled."
                        fi
                    fi
                fi
            fi
            ;;
        Q)
            echo "Aborting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Aborting..."
            exit 0
            ;;
    esac
done
