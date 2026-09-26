from processing_module import build_preprocessed_windows_dataset, save_site_in_chunks

print("Starting the eeg processing pipeline...")
exams_after_eda_processing = 51135
ds = build_preprocessed_windows_dataset(idxs = [i for i in range(exams_after_eda_processing)])
site_datasets = ds.split(by="site")

for site_name, site_dataset in site_datasets.items():

    save_site_in_chunks(
        site_ds=site_dataset,
        site_name=site_name)
