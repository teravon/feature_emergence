def get_dataset_filepath(dataset_root_folder, dataset_name, npoi, leakage_model):
    if leakage_model == "ID":
        dataset_dict = {
            "ascad-variable": {
                1400: f"{dataset_root_folder}/ascad-variable.h5",
                
                2000: f"{dataset_root_folder}/ASCADr/ascad-variable_70k-90k_20.h5",
                
                250000: f"{dataset_root_folder}/ASCADr/atmega8515-raw-traces.h5",
            },
            "eshard": {
                1400: f"{dataset_root_folder}/eshard.h5",
            },
        }
    else:
        dataset_dict = {
          
            "ascad-variable": {
                # 100: f"{dataset_root_folder}/ASCADr/ascad-variable_rpoi/ascad-variable_100poi_hw.h5",
                1400: f"{dataset_root_folder}/ascad-variable.h5",
                2000: f"{dataset_root_folder}/ASCADr/ascad-variable_70k-90k_20.h5",
                
                250000: f"{dataset_root_folder}/ASCADr/atmega8515-raw-traces.h5",
            },
            "eshard": {
                1400: f"{dataset_root_folder}/eshard.h5",
            },
        
        }
    return dataset_dict[dataset_name][npoi]
