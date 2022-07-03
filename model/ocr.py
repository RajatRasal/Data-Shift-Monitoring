import easyocr


def load_model():
    return easyocr.Reader(
        lang_list=["en"],
        download_enabled=True,
        # model_storage_directory=models_dir,
    )
