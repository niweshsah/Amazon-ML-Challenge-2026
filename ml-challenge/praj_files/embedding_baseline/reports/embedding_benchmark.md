# Embedding benchmark

Sample records: 100,000

| Text | Dim | Length | rows/s | tokens/s | Full target ETA (h) |
|---|---:|---:|---:|---:|---:|
| name_address_country | 512 | 64 | 542.6 | 18912.0 | 5.28 |
| name_address_country | 256 | 64 | 539.4 | 18772.5 | 5.32 |
| name_only | 512 | 64 | 565.6 | 5685.1 | 5.07 |
| name_only | 256 | 64 | 434.3 | 4351.6 | 6.60 |

Quantization was not tested because no verified local quantized model/runtime was available.
