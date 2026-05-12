from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def download():
    print("--- Pre-downloading FastEmbed Model (ONNX) ---")
    FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("--- Download Complete ---")


if __name__ == "__main__":
    download()
