#!/bin/bash

# 実行したいフォルダのリスト
dirs=("pz_0.0" "pz_0.05" "pz_0.1" "pz_0.15" "pz_0.2")

# 各フォルダで逐次実行
for dir in "${dirs[@]}"; do
    echo "=== Running in $dir ==="
    cd "$dir"

    # mpirun のコマンド（必要に応じて-nの値などを調整）
    mpirun -n 20 python triangular_EE_initial_Z_mpi.py

    cd ..
    echo "=== Done with $dir ==="
done
