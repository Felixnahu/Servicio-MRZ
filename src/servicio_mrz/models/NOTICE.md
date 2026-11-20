# Modelos incluidos

## docaligner_lcnet100_h_e_bifpn_256_fp32.onnx

- Origen: [DocAligner](https://github.com/DocsaidLab/DocAligner) (DocsaidLab), modelo
  `lcnet100_h_e_bifpn_256_fp32` (PP-LCNet-1.0 + BiFPN, regresion por mapa de calor de las 4
  esquinas de un documento).
- Licencia: Apache License 2.0 (codigo y pesos publicados por DocsaidLab).
- Copiado sin modificar desde `kyc-ci-backend` (`src/docproc/engines/models/`).
- Entrada: `1x3x256x256` BGR float32 en `[0, 1]`. Salida `heatmap`: `1x4x128x128`.
- SHA-256: `f4117b786e3a18470f3865c93f3c2bd69d9b998edd60f385574a5c665e79594e`
  (se verifica al cargar en `cardfinder.py`).
