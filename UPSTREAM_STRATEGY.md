# Stratégie de contribution upstream (huggingface/candle)

Retour du mainteneur, août 2026 : trop de PR, trop grosses, descriptions générées,
et des crates qui n'existent que dans le fork. Ce document mesure le problème,
tranche PR par PR, et fixe les règles pour la suite.

Mesures faites en local contre `upstream/main` (`2a13b0f`), via les refs
`refs/pull/<n>/head`. Tous les chiffres ci-dessous sont vérifiables :

```sh
git remote add upstream https://github.com/huggingface/candle
git fetch upstream main 'refs/pull/*/head:refs/remotes/upr/*'
git diff --numstat $(git merge-base upstream/main upr/3770) upr/3770 | wc -l
```

## 1. L'état réel

| Mesure | Valeur |
| --- | --- |
| PR ouvertes (auteur `astorise`) | 41 |
| PR qui ajoutent des crates absentes d'upstream | **16** |
| Grappes de doublons | **8** |
| PR réellement isolées et défendables | **6** |
| Plus grosse PR | #3770 — 15 016 lignes, 109 fichiers, 148 commits |

Les crates concernées : `candle-nvfp4-kernels`, `candle-fp8-kernels`,
`candle-awq-kernels`, `candle-gptq-kernels`, `candle-flashinfer-kernels`,
`candle-lora-kernels`. Aucune n'existe upstream. C'est exactement le point
soulevé par le mainteneur, et il concerne 39 % des PR ouvertes.

## 2. Les grappes de doublons

C'est le problème le plus coûteux pour le mainteneur, et le plus facile à corriger :
à chaque révision, une nouvelle PR a été ouverte au lieu de pousser sur la branche
existante.

| Sujet | PR | Constat |
| --- | --- | --- |
| NVFP4 | 3825, 3833, 3849, 3858, 3861, 3883 | Les 4 dernières ont le **même** ensemble de fichiers, `nvfp4-kernels/src/lib.rs` figé à 399 lignes |
| Qwen 3.5 | 3821, 3834, 3838, 3869 | Mêmes 6 fichiers ; `qwen3_5.rs` grossit 764 → 1182 → 1496 → 1731 lignes |
| Snapshots du fork | 3726 ⊂ 3733 ⊂ 3770 | Inclusion git stricte : 79 → 93 → 109 fichiers, 55 → 65 → 148 commits |
| CUDA graph | 3669, 3729, 3733 | Mêmes `cuda_backend/graph.rs` + `tests/cuda_graph_tests.rs` |
| LoRA | 3689, 3762, 3763, 3767, 3770 | Même pile, ré-empilée à chaque fois |
| IncrementalDecoder | 3789, 3811 | Diff entre les deux : 29 lignes ajoutées, 99 supprimées, 1 fichier |
| Quantification | 3660, 3661, 3662 | Chacune ré-ajoute les crates kernels de la précédente |
| CUDA récent | 3885 ⊂ 3895, 3894 ⊂ 3895 | #3895 empaquette 5 changements sans rapport |

Vu de la file d'attente du mainteneur, 41 PR représentent en réalité **une douzaine
d'idées distinctes**. Le reste est du bruit de révision.

## 3. Décisions

### 3.1 Garder — 6 PR, dans cet ordre, une à la fois

Classées par probabilité de merge. **N'ouvrir la suivante que quand la précédente est
mergée ou fermée.**

| # | PR | Taille | Pourquoi elle passe |
| --- | --- | --- | --- |
| 1 | **#3894** conv2d im2col | 1 fichier, +5/−4 | Vrai bug, diff minimal, évident à relire |
| 2 | **#3632** Metal SDPA mask+causal | 3 fichiers | Vrai NaN, test inclus, périmètre net |
| 3 | **#3787** `Device::supports_qmatmul` | 2 fichiers, +61 | Petite API + test qui la vérifie |
| 4 | **(nouvelle)** nommage f8e4m3 | ~10 fichiers, mécanique | Bug systématique, voir §3.2 |
| 5 | **#3885** erreur kernel manquant | 4 fichiers | DX pure — retirer le bump cudarc |
| 6 | **#3765** saturation f16/bf16 | 8 fichiers | Suite de #3717 déjà mergée |

En fin de file, si les six passent : **#3647** (propagation du device dans
`candle-onnx`). Vrai problème — `simple_eval` force le CPU — mais 8 fichiers et ça
touche PyO3. La laisser ouverte sans la pousser.

### 3.2 Le bug f8e4m3 mérite sa propre PR

C'est la trouvaille la plus solide du lot, et elle est enterrée dans #3895.

Sur `upstream/main` :

- `DType::F8E4M3.as_str()` vaut `"f8e4m3"` (`candle-core/src/dtype.rs:87`)
- donc `kernel_name::<T>(root)` génère `ucopy_f8e4m3`, `affine_f8e4m3`, `cast_f8e4m3_f32`…
- mais les `.cu` définissent `ucopy_f8_e4m3`, `cast_f8_e4m3_f32`, `const_set_f8_e4m3`

**Tous les kernels CUDA F8E4M3 sont donc introuvables à l'exécution.** Le cas le plus
direct : `copy_strided_src` sur un tenseur F8E4M3 non contigu
(`cuda_backend/mod.rs:2498`) échoue systématiquement.

À envoyer seule, sans le fix conv1d, sans le fix conv2d, sans le changement de
message d'erreur, sans le bump cudarc. #3895 est à fermer.

### 3.3 Fermer maintenant — 35 PR

41 ouvertes − 6 gardées = 35 à fermer. Trois motifs, un seul par PR, pas de
justification longue.

**A — embarque des crates absentes d'upstream** (16) :
3660, 3661, 3662, 3669, 3726, 3733, 3763, 3767, 3770, 3825, 3833, 3834, 3849,
3858, 3861, 3883

**B — doublon d'une PR de la même série** (7, voir §2) :
3789, 3811, 3821, 3838, 3869, 3729, 3895

**C — trop large pour un premier contact** (12) :
3648, 3657, 3689, 3692 (DeepSeek-V3), 3693 (Llama 4), 3721, 3762, 3794
(dispatcher GGUF), 3818, 3827 (JSON Schema, 2 734 lignes), 3829
(tensor-parallel), 3867 (ModelOptCheckpoint)

Un commentaire d'une ligne suffit à chaque fermeture. Ne pas argumenter, ne pas
demander de réévaluation. La fermeture *est* le message.

### 3.4 Externaliser

Le mainteneur l'a demandé explicitement. Rien de tout ça n'exige de modification
d'upstream : `CustomOp1/2/3`, le trait `Module` et `VarBuilder` suffisent à brancher
des kernels et des couches depuis une crate tierce.

| Crate à créer | Contenu | PR remplacées |
| --- | --- | --- |
| `tachyon-nvfp4` | `candle-nvfp4-kernels` + `quantized_nvfp4` | 3825, 3833, 3849, 3858, 3861, 3883 |
| `tachyon-lora` | kernels BGMV, `LoraLinear`, batching hétérogène | 3689, 3762, 3763, 3767, 3770 |
| `tachyon-quant` | GPTQ / AWQ / FP8 par blocs | 3660, 3661, 3662 |
| `tachyon-attn` | FlashInfer, CUDA graph, paged attention CPU/Metal | 3657, 3669, 3721, 3726, 3729, 3733 |
| `tachyon-models` | qwen3_5, deepseek_v3, llama4 | 3692, 3693, 3821, 3834, 3838, 3869 |
| `tachyon-decoding` | contraintes JSON Schema, `IncrementalDecoder` | 3789, 3811, 3827 |

Bénéfice direct pour Tachyon : ces crates cessent d'être bloquées par le rythme de
revue d'upstream, et le fork peut suivre `main` sans rebase permanent.

## 4. Règles pour la suite

1. **Une PR = un concept = idéalement un commit.** Si le titre contient « et » ou
   « + », la PR est à découper.
2. **Jamais de deuxième PR sur le même fichier.** Une révision se pousse sur la
   branche existante.
3. **Description : 5 lignes maximum.** Ce qui casse, comment le reproduire, ce qui
   change. Pas de titres de section, pas de listes à puces, pas de gabarit
   « Summary / Motivation / Testing ». Si ça se lit comme un rapport, c'est trop long.
4. **Brancher sur `upstream/main`**, jamais sur `main` du fork.
5. **Zéro référence à une crate qui n'existe pas upstream.** À vérifier avant push :
   `git diff --name-only upstream/main... | grep -E 'candle-(fp8|awq|gptq|flashinfer|lora|nvfp4)-kernels'`
   doit ne rien retourner.
6. **Une PR en vol à la fois** tant que la confiance n'est pas rétablie.
7. **Bug d'abord, fonctionnalité ensuite.** Un correctif de 5 lignes avec un test
   achète plus de crédit que 3 000 lignes de modèle.

## 5. Textes prêts à envoyer

### 5.1 Réponse au mainteneur

> You're right, and thanks for being direct about it.
>
> I've closed 35 of my open PRs: everything that carried kernel crates that only
> exist in my fork, plus several series where I'd opened a new PR instead of
> updating the existing branch. That work is moving to separate crates outside
> candle.
>
> I've kept five, each isolated, rewritten short: #3894, #3632, #3787, #3885,
> #3765. I won't open anything new until those are resolved either way.
>
> One thing worth flagging separately: `candle-core` looks up `ucopy_f8e4m3` while
> `unary.cu` defines `ucopy_f8_e4m3`, and `DType::as_str()` returns `f8e4m3`, so
> every F8E4M3 CUDA kernel is unreachable. I can send that as a standalone rename
> if it's useful.

### 5.2 Commentaire de fermeture (motif A)

> Closing — this carries kernel crates that don't exist upstream. Moving it to a
> separate crate.

### 5.3 Commentaire de fermeture (motif B)

> Closing as a duplicate of #<n>, which covers the same files.

### 5.4 Commentaire de fermeture (motif C)

> Closing — too broad for now. I'll revisit as a smaller, isolated change if
> there's interest.

### 5.5 Descriptions réécrites

**#3894 — Fix conv2d CUDA im2col offset for non-contiguous kernels**

> On CUDA, `conv2d` with a non-contiguous kernel copies the kernel into a fresh
> contiguous buffer `kernel_c`, then calls `matmul` with the *original* `kernel`
> and a layout carrying the original start offset. The materialized copy is never
> read.
>
> Pass `kernel_c`, with the layout at offset 0.

**#3632 — Metal SDPA: allow an additive mask together with causal masking**

> `call_sdpa_full` clears `do_causal` whenever a mask is supplied, so a padding or
> bias mask combined with causal masking attends to future keys. The guard exists
> because a fully-masked query row ends with `sum_score == 0` and the final divide
> yields NaN.
>
> Clamp a zero normalizer to 1 in the kernel so such a row stays zeroed, and let
> both inputs apply. Test in `candle-nn/tests/sdpa.rs`.

**#3787 — Add `Device::supports_qmatmul`**

> There's no way to ask a device whether it can run `QMatMul` for a given
> `GgmlDType` short of running it and catching the error. `Q8_1` and `Q8K` are
> activation formats: CUDA has no matmul kernel for them, Metal only handles them
> at `m == 1`.
>
> Adds `Device::supports_qmatmul(GgmlDType) -> bool` plus a test asserting it
> agrees with the CPU backend across all 15 dtypes.

**(nouvelle) — candle-kernels: name F8E4M3 kernels `f8e4m3`**

> `DType::F8E4M3.as_str()` is `"f8e4m3"`, so `kernel_name::<T>()` generates lookups
> like `ucopy_f8e4m3` and `cast_f8e4m3_f32`, while the `.cu` sources define
> `ucopy_f8_e4m3` and `cast_f8_e4m3_f32`. Every F8E4M3 CUDA kernel is unreachable —
> `copy_strided_src` on a non-contiguous F8E4M3 tensor fails today.
>
> Renames the CUDA symbols to the `f8e4m3` spelling and updates the one hardcoded
> Rust string still using the old form.

**#3885 — Name the missing kernel and module in CUDA symbol-lookup errors**

> A missing CUDA symbol reports `missing kernel '<module>'`: it names the module
> but not the kernel, and discards the driver error.
>
> `CudaError::MissingKernel` now carries the kernel name, the module name and the
> `DriverError` as `#[source]`; `kernels::Module` gains `name()`.

*(retirer le bump `cudarc` 0.19.8 → 0.19.9 de cette PR — sans rapport)*

**#3765 — Accumulate f16/bf16 reductions in f32**

> f16/bf16 reductions accumulate in the storage dtype, so a `sum` over a large
> non-trailing axis, `avg_pool2d` over a large window, and the softmax denominator
> saturate to inf before the result is cast back.
>
> Accumulate those three in f32 and cast once at the end. Follows #3717, which
> covered the trailing-axis case.
