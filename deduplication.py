import os
from concurrent.futures.process import ProcessPoolExecutor
from rapidfuzz import fuzz
from datasketch import MinHash, MinHashLSH
import json
import argparse
import shutil


def get_shingles(text, k: int = 20):
    """AI is creating summary for get_shingles

    Args:
        text (string): text from document
        k (int, optional): Defaults to 20, size of shingles to create minhash

    Returns:
        shinges (list): list of shingles to create minhash
    """
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add("".join(text[i : i + k]))
    return shingles


def create_minhash(text, num_perm: int = 128):
    """

    Args:
        text (list): text shingles
        num_perm (int): number of perumation to create minhash

    Returns:
        hash (Minhash): min hash of text shingles
    """
    hash = MinHash(num_perm=num_perm)
    shingles = get_shingles(text)
    for s in shingles:
        hash.update(s.encode("utf8"))
    return hash


def delete_empty_files(dir):
    """after preprocessing some document maybe empty, remove empty document before deduplication to reduce redundancy

    Args:
        dir (str): directory of documents chunk
    """
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path) and os.stat(file_path).st_size == 0:
            os.remove(file_path)
            #change directory if required
            with open("../dedup_outputs/empty_file.txt", "a") as file:
                file.write(f"{filename}\n")
            print(f"Deleted empty file: {file_path}")


def find_near_duplicates(folder_path, threshold, num_perm):
    """AI is creating summary for find_near_duplicates

    Args:
        folder_path (str): path to document chunk folder
        threshold (int): min hash threshold
        num_perm (int): number of permutation

    Returns:
        duplicates (dict): dictionary object that store potential duplicate file
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    document_signature = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            text = text.split(" ")
            m = create_minhash(text, num_perm)
            lsh.insert(filename, m)
            document_signature[filename] = m
    duplicate = {}

    for filename, minhash in document_signature.items():
        results = lsh.query(minhash)

        if len(results) > 1:
            duplicate[filename] = results
    print("finish finding duplicate in min hash")
    return duplicate


def fuzzy_search(doc1_path, doc2_path):
    """

    Args:
        doc1_path (str): path for document 1
        doc2_path (str): path for document 2

    Returns:
        ration (int): similarity ration between document 1 and document 2
    """
    try:
        with open(doc1_path, "r", encoding="utf-8") as f:
            doc1 = f.read()
        with open(doc2_path, "r", encoding="utf-8") as f1:
            doc2 = f1.read()
        ratio = fuzz.ratio(doc1, doc2)
        return ratio
    except FileNotFoundError:
        print("file has been removed")


def efficient_fuzzy_search(
    folder_path,
    minhash_threshold=0.5,
    fuzzy_threshold=60,
    num_perm=128,
    save_json=True,
    output_filename=None,
):
    """Function to perform duplicaiton detection using both minhash and fuzzysearch with parallel processing

    Args:
        folder_path (str): directories with documents
        output_filename (str)): name of the output file
        minhash_threshold (float, optional): the min hash similarity threshold. Defaults to 0.5.
        fuzzy_threshold (int, optional): the near similarity score for string comparison using fuzzy search. Defaults to 60.
        num_perm (int, optional): permutation of minhash. Defaults to 128.

    Returns:
        similar doc(dict): dictionary object that store similar documents and similar threshold
    """

    near_duplicates = find_near_duplicates(
        folder_path, threshold=minhash_threshold, num_perm=num_perm
    )

    similar_docs = {}
    count = 0
    output_files_path =[]
    with ProcessPoolExecutor() as executor:
        file_count = 0
        for doc1, potential_duplicates in near_duplicates.items():
            doc1_path = os.path.join(folder_path, doc1)
            count += 1
            for doc2 in potential_duplicates:
                doc2_path = os.path.join(folder_path, doc2)
                if doc1 != doc2:
                    similarity = executor.submit(
                        fuzzy_search, doc1_path, doc2_path
                    ).result()
                    if similarity is not None:
                        if similarity >= fuzzy_threshold:
                            if doc1 not in similar_docs:
                                similar_docs[doc1] = []
                            similar_docs[doc1].append((doc2, similarity))

        if similar_docs:
            if save_json and output_filename is not None:
                output_folder_path = "dedup_outputs"
                name = output_filename + "_" + str(file_count) + ".json"
                output_path = os.path.join(output_folder_path, name)
                output_files_path.append(name)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(similar_docs, f, indent=4)
                print(f"Similar documents written to {output_filename}")
                print(output_path)
    return similar_docs,output_files_path


def collection_and_remove_duplicate(dir, threshold,output_dir="dedup_outputs",file=None,track=True):
    """AI is creating summary for collection_and_remove_duplicate

    Args:
        dir (string): directory that store json file that contain document duplicate information
        file (string): output to store the report of deduplication
        threshold (int): threshold to remove duplicate document
    """
    if file:
        file_path = os.path.join(output_dir,file)
        with open(file_path, "r") as f:
            dup_info = dict(json.load(f))
    else:
        dup_info = efficient_fuzzy_search(
            minhash_threshold=0.5, fuzzy_threshold=threshold, num_perm=128)

    if track is True:
        track_dir = "dedup_outputs"
        track_fn = "dedup_track.txt"
        track_path = os.path.join(track_dir, track_fn)

    for doc in dup_info.keys():
        longest_doc = doc

        for d, score in dup_info.get(doc, []):
            cur_path = os.path.join(dir, d)
            longest_doc_path = os.path.join(dir, longest_doc)

            length = os.path.getsize(cur_path) if os.path.exists(cur_path) else 0
            longest_doc_length = (
                os.path.getsize(longest_doc_path)
                if os.path.exists(longest_doc_path)
                else 0
            )

            if score >= threshold:
                if length > longest_doc_length and score > threshold:
                    if os.path.exists(longest_doc_path):
                        os.remove(longest_doc_path)
                        if track is True:
                            with open(track_path, 'a') as file:
                                file.write(f'{longest_doc}\n')
                    longest_doc = d
                else:
                    if os.path.exists(cur_path) and score > threshold:
                        os.remove(cur_path)
                        if track is True:
                            with open(track_path, 'a') as file:
                                file.write(f'{longest_doc}\n')

def main():
    parser = argparse.ArgumentParser(description="Perform fuzzy search and deduplicate files in a directory.")
    parser.add_argument("folder_path", help="Path to the folder containing documents")
    parser.add_argument("--minhash_threshold", type=float, default=0.5, help="MinHash similarity threshold")
    parser.add_argument("--fuzzy_threshold", type=int, default=60, help="Fuzzy matching threshold")
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations for MinHash")
    parser.add_argument("--save_json", action="store_true", help="Save results to a JSON file")
    parser.add_argument("--output_filename", type=str, default="dap", help="Name of the output JSON file")
    parser.add_argument("--dedup_threshold", type=int, default=80, help="Threshold for duplicate removal")
    parser.add_argument("--output_dir", type=str, default="dedup_outputs", help="Directory to save deduplicated files")

    args = parser.parse_args()

    # Ensure folder path exists
    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a valid directory.")
        return

    # Run fuzzy search
    similar_doc, files = efficient_fuzzy_search(
        folder_path=args.folder_path,
        minhash_threshold=args.minhash_threshold,
        fuzzy_threshold=args.fuzzy_threshold,
        num_perm=args.num_perm,
        save_json=args.save_json,
        output_filename=args.output_filename
    )


    # Process files for deduplication
    for file in files:
        collection_and_remove_duplicate(
            dir=args.folder_path,
            threshold=args.dedup_threshold,
            file=file,
            output_dir=args.output_dir,
            track=True
        )
if __name__ == '__main__':
    """
    Example command to execute:
    python deduplication.py "/home/uykuite/Documents/Data Source/article_dap/test" \
    --minhash_threshold 0.5 \
    --fuzzy_threshold 80 \
    --num_perm 128 \
    --save_json \
    --output_filename "dap" \
    --dedup_threshold 80 \
    --output_dir "dedup_outputs"
    """
    main()
