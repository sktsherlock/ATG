from huggingface_hub import upload_file

upload_file(path_or_fileobj="Movies_convnextv2_huge.npy", path_in_repo="Movies/ImageFeature/Movies_convnextv2_huge.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_swinv2_large.npy", path_in_repo="Movies/ImageFeature/Movies_swinv2_large.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_vit_large.npy", path_in_repo="Movies/ImageFeature/Movies_vit_large.npy", repo_id="Sherirto/MAG")