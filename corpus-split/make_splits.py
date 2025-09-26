import argparse, hashlib, zipfile, pathlib
import numpy as np
import pandas as pd

def sha256_of_file(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def write_parallel(df: pd.DataFrame, prefix: str):
    cols = ['id','ru','ko']
    df[cols].to_csv(f"{prefix}.csv", index=False, encoding="utf-8")
    df['ru'].to_csv(f"{prefix}.ru", index=False, header=False, encoding="utf-8", lineterminator="\n")
    df['ko'].to_csv(f"{prefix}.ko", index=False, header=False, encoding="utf-8", lineterminator="\n")

def main(csv_path: str, seed: int, version: str):
    csv_path = str(pathlib.Path(csv_path).resolve())
    df = pd.read_csv(csv_path, encoding="utf-8")
    assert {'id','ru','ko'}.issubset(df.columns), "CSV must have id, ru, ko"
    n = len(df)
    print(f"[INFO] Loaded: {csv_path} (n={n})")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(round(n*0.8))
    n_dev   = int(round(n*0.1))
    n_test  = n - n_train - n_dev

    train = df.iloc[idx[:n_train]].reset_index(drop=True)
    dev   = df.iloc[idx[n_train:n_train+n_dev]].reset_index(drop=True)
    test  = df.iloc[idx[n_train+n_dev:]].reset_index(drop=True)

    # leakage check (id 기준)
    assert set(train['id']).isdisjoint(dev['id'])
    assert set(train['id']).isdisjoint(test['id'])
    assert set(dev['id']).isdisjoint(test['id'])

    print(f"[INFO] Split sizes  train={len(train)}  dev={len(dev)}  test={len(test)}")

    write_parallel(train, "train")
    write_parallel(dev,   "dev")
    write_parallel(test,  "test")

    files = ["train.csv","dev.csv","test.csv","train.ru","train.ko","dev.ru","dev.ko","test.ru","test.ko"]

    with open("CHECKSUMS.sha256","w",encoding="utf-8") as w:
        for fn in files:
            w.write(f"{sha256_of_file(fn)}  {fn}\n")

    zipname = f"splits_{version}.zip"
    with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files + ["CHECKSUMS.sha256"]:
            z.write(fn)

    arch_hash = sha256_of_file(zipname)
    with open("ARCHIVE_SHA256.txt","w",encoding="utf-8") as w:
        w.write(f"{arch_hash}  {zipname}\n")

    print(f"[DONE] {zipname}")
    print(f"[SHA-256] {arch_hash}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v1.0")
    a = p.parse_args()
    main(a.csv, a.seed, a.version)
