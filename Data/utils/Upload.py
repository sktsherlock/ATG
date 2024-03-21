from huggingface_hub import upload_file

upload_file(path_or_fileobj="Movies_gemma_7b_256_mean.npy", path_in_repo="Movies/TextFeature/Movies_gemma_7b_256_mean.npy", repo_id="Sherirto/MAG")