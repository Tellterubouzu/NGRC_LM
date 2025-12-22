
# 研究計画 DeepNGRC
- [ ] 目標 : GPTよりも軽量に学習，推論が可能なモデルを匹敵する性能で作成する

- [ ] 新規性/アイデア
    - [ ] NGRCにSelective Copyingの概念を導入し言語モデルを作成
    - [ ] NGRCの特徴生成部分の演算を共通化，low-rankな調整でlayerごとの演算を行う？
- [ ] 有効性　(他の言語モデルアーキテクチャと比べて)
    - [ ] 推論時の計算量 (n 倍)
    - [ ] 学習の速度 (n 倍)
    - [ ] スケーリングカーブ (n 倍)
    - [ ] 下流タスクの性能 (n 倍)
- [ ]　先行研究
    - [ ] ESN
    - [ ] NGRC
        - [ ] NGRC
            - [ ] Next Generation Reservoir Computing
        - [ ] Multilayered-NGRC models
            - [ ] Layer Varying Deep Reservoir Computing Architecture
            - [ ] Next-Generation Reservoir Computing for Dynamical Inference
            - [ ] Adaptive Nonliniear Vector Autoreguression - Robust Forecasting for Noisy Chaotic Time Series
            - [ ] Temporal Convolution Dericed Multi-layer Reservoir Computing

    - [ ] 軽量化
        - [ ] Attention系
            - [ ] Linear Attention
            - [ ] Grouped Query Attention
            - [ ] Multihead Latetent Attention
            - [ ] Deepseek Sparse Attention
        - [ ] RNN系列
            - [ ] RWKV
            - [ ] LesNet
        - [ ] SSM系
            - [ ] Mamba
            - [ ] Mamba-2
            - [ ] Mamba-3
- [ ] 技術的な課題
    - [ ] Multi node 学習 (tensor並列が遅くなるため，pipeline並列にする必要がある)
    - [ ] ハードウェア（HBM, L1/L2メモリを意識した高速化）
    [compute intensity](./images/computeintensity.png)

- [ ] これまでの知見 
    - [ ] 50M 前後の実験では多層のGPTにNGRCのperplexityが匹敵
    - [ ] 100M 150MTokenの学習で多層のGPTが上回るため，NGRCも多層化が必要

- [ ] アブレーションスタディ
    - [ ] 単層NGRC
    - [ ] 特徴の非線形性について poly
    - [ ] スケーリング
    - [ ] Selective Copying(ゲート)
    - [ ] 演算の共通化

- [ ] 比較実験
    - [ ] GPT-2
    - [ ] Transformer++
    - [ ] Mamba-3
    - [ ] LinearAttention
    - [ ] RWKV
    - [ ] Deepseek Sparse Attention

- [ ] 評価ベンチマークテスト
    - [ ] mambaで使われてるようなやつ


