

PRETRAINED_MODELS_PATHS = {
    "SaProt-35M": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-SaProt-35M-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-SaProt-35M.pth",
        "stats": "checkpoints/statistics/encodings-SaProt-35M.pth"
    },
    "ESM2-3B": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-ESM2_3B-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-ESM2-3B.pth",
        "stats": "checkpoints/statistics/encodings-ESM2-3B.pth"
    },
    "ESM2-8M": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-ESM2_8M-AFDB/1000000.pth",
        "decoder": "checkpoints/decoder_checkpoints/transformer-decoder-ESM2-8M.pth",
        "stats": "checkpoints/statistics/encodings-ESM2-8M.pth"
    },
    "CHEAP_shorten_1_dim_1024": {
        "diffusion": "checkpoints/diffusion_checkpoints/DiMA-bert_35M-CHEAP_shorten_1_dim_1024-AFDB/500000.pth",
        "stats": "checkpoints/statistics/encodings-CHEAP_shorten_1_dim_1024.pth"
    }
}