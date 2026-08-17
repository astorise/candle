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
| Séries redondantes | **8** |
| PR isolées et défendables | **6** (1 ouverte, 5 à resoumettre) |
| PR gardée pour collaboration active | **1** — #3838, voir §3.6 |
| Plus grosse PR | #3770 — 15 016 lignes, 109 fichiers, 148 commits |

Les crates concernées : `candle-nvfp4-kernels`, `candle-fp8-kernels`,
`candle-awq-kernels`, `candle-gptq-kernels`, `candle-flashinfer-kernels`,
`candle-lora-kernels`. Aucune n'existe upstream. C'est exactement le point
soulevé par le mainteneur, et il concerne 39 % des PR ouvertes.

## 2. Les séries redondantes

C'est le problème le plus coûteux pour le mainteneur : à chaque révision, une nouvelle
PR a été ouverte au lieu de pousser sur la branche existante. Attention au vocabulaire —
seules les lignes marquées `⊂` sont des inclusions strictes vérifiées ; les autres sont
des branches **divergentes** sur les mêmes fichiers (voir §3.4).

| Sujet | PR | Constat |
| --- | --- | --- |
| NVFP4 | 3825, 3833, 3849, 3858, 3861, 3883 | Les 4 dernières ont le **même** ensemble de fichiers, `nvfp4-kernels/src/lib.rs` figé à 399 lignes |
| Qwen 3.5 | 3821, 3834, 3838, 3869 | Mêmes 6 fichiers ; `qwen3_5.rs` grossit 764 → 1182 → 1496 → 1731 lignes |
| Snapshots du fork | 3726 ⊂ 3733 ⊂ 3770 | Inclusion git stricte : 79 → 93 → 109 fichiers, 55 → 65 → 148 commits |
| CUDA graph | 3669, 3729, 3733 | Mêmes `cuda_backend/graph.rs` + `tests/cuda_graph_tests.rs` |
| LoRA | 3689, 3762, 3763, 3767, 3770 | Même pile, ré-empilée à chaque fois |
| IncrementalDecoder | 3789, 3811 | Diff entre les deux : 29 lignes ajoutées, 99 supprimées, 1 fichier |
| Quantification | 3660, 3661, 3662 | Chacune ré-ajoute les crates kernels de la précédente |
| CUDA récent | 3885, 3894, 3895 | #3895 réimplémente les deux et empaquette 5 changements sans rapport |

Vu de la file d'attente du mainteneur, 41 PR représentent en réalité **une douzaine
d'idées distinctes**. Le reste est du bruit de révision.

## 3. Décisions

### 3.0 État au 17 août 2026 — deux PR ouvertes

Les fermetures sont faites. Il reste **#3894** (prévu) et **#3838** (gardée en plus,
parce qu'elle a un collaborateur actif — voir §3.6). Tout le reste est fermé.

### 3.1 Garder une seule PR ouverte : #3894

Une file de 6 PR reste une file. Après un retour qui dit « quality over quantity »,
la seule réponse lisible est **une PR ouverte, une seule**.

**#3894** — conv2d im2col, 1 fichier, +5/−4. C'est le plus petit ticket d'entrée
possible : un bug réel, un diff qu'on relit en dix secondes, aucune dépendance.
Réécrire sa description (§5.6) et ne rien pousser d'autre.

Fermer ne veut pas dire abandonner : les branches restent dans le fork, le travail
est fait. Les cinq autres se resoumettent une par une, **chacune seulement quand la
précédente est mergée ou fermée** :

| Ordre | PR | Taille | Action |
| --- | --- | --- | --- |
| 1 | **#3894** conv2d im2col | 1 fichier, +5/−4 | **Reste ouverte**, description réécrite |
| 2 | nommage f8e4m3 | ~10 fichiers, mécanique | Nouvelle PR, extraite de #3895 — voir §3.2 |
| 3 | #3632 Metal SDPA mask+causal | 3 fichiers | Resoumettre |
| 4 | #3787 `Device::supports_qmatmul` | 2 fichiers, +61 | Resoumettre |
| 5 | #3885 erreur kernel manquant | 4 fichiers | Resoumettre sans le bump cudarc |
| 6 | #3765 saturation f16/bf16 | 8 fichiers | Resoumettre |

**#3647** (propagation du device dans `candle-onnx`) : vrai problème — `simple_eval`
force le CPU — mais 8 fichiers et ça touche PyO3. Fermer aussi, et ne la reconsidérer
qu'une fois les six passées.

À ce rythme, six correctifs mergés valent mieux que quarante PR ouvertes. Et si le
premier passe, le deuxième se lit avec un a priori favorable.

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

### 3.3 Fermées — 39 PR

41 ouvertes − 2 gardées (#3894, #3838) = 39 fermées. Quatre motifs, un seul par PR,
pas de justification longue.

**A — embarque des crates absentes d'upstream** (16) :
3660, 3661, 3662, 3669, 3726, 3733, 3763, 3767, 3770, 3825, 3833, 3834, 3849,
3858, 3861, 3883

**B — série redondante** (6, voir §2 et §3.4) :
3729, 3789, 3811, 3821, 3869, 3895

*(#3838 appartenait à cette série mais reste ouverte — voir §3.6.)*

**C — trop large pour un premier contact** (12) :
3648, 3657, 3689, 3692 (DeepSeek-V3), 3693 (Llama 4), 3721, 3762, 3794
(dispatcher GGUF), 3818, 3827 (JSON Schema, 2 734 lignes), 3829
(tensor-parallel), 3867 (ModelOptCheckpoint)

**D — bonnes, mais à resoumettre plus tard** (5) :
3632, 3647, 3765, 3787, 3885

Un commentaire d'une ligne suffit à chaque fermeture. Ne pas argumenter, ne pas
demander de réévaluation. La fermeture *est* le message.

Fait le 17 août 2026.

### 3.4 Le mot « doublon » est trop fort — et c'est pire que ça

Vérification faite (`git merge-base --is-ancestor` + diff bidirectionnel entre chaque
paire), **aucune PR du motif B n'est un sur-ensemble ni un ancêtre git d'une autre.**
Ce ne sont pas des révisions successives d'une même branche : ce sont des branches
**divergentes** sur les mêmes fichiers.

| Paire | Diff X → Y | Containment |
| --- | --- | --- |
| 3838 → 3869 | +246 / −11, 2 fichiers | non (le plus proche) |
| 3789 → 3811 | +29 / −99, 1 fichier | non |
| 3729 → 3733 | +10 261 / −240, 87 fichiers | non |
| 3821 → 3838 | +808 / −670, 4 fichiers | non |
| 3894 → 3895 | +272 / −117, 18 fichiers | non |

Conséquence pratique : **il n'existe pas de version canonique** de Qwen 3.5, du CUDA
graph ni de l'IncrementalDecoder. Quatre PR Qwen 3.5, quatre `qwen3_5.rs` différents,
et #3821 est la seule à porter `quantized_lm.rs`. C'est un problème plus sérieux qu'un
simple doublon : avant d'externaliser, il faut choisir une version de référence par
sujet et jeter les autres.

Conséquence pour les commentaires de fermeture : **ne citer aucune PR sœur**, puisque
toutes ferment aussi. Envoyer le mainteneur vers une PR fermée est pire que de ne rien
dire. Les textes exacts sont en §5.3.

### 3.5 Externaliser — après vérification de l'écosystème

Le mainteneur l'a demandé explicitement. Rien de tout ça n'exige de modification
d'upstream : `CustomOp1/2/3`, le trait `Module` et `VarBuilder` suffisent à brancher
des kernels et des couches depuis une crate tierce. Le précédent existe chez HF
eux-mêmes : `huggingface/candle-flash-attn-v1` et `michaelfeil/candle-flash-attn-v3`
sont des dépôts séparés.

**Vérification faite le 17 août 2026** (crates.io + GitHub). Les six noms sont libres
sur crates.io, mais ça ne dit rien de l'existant :

| Dépôt créé | crates.io | Collision / concurrent établi | Verdict |
| --- | --- | --- | --- |
| `candle-nvfp4` | libre | `float4` (700 k dl) fait MXFP4, pas NVFP4 ; mistral.rs ne fait pas NVFP4 | **Garder** — seul créneau réellement ouvert |
| `candle-lora` | libre | **`EricLBuehler/candle-lora` 175 ⭐, même nom, même objet, maintenu (juil. 2026)** ; + `jammi-lora` publié ; + LoRA/X-LoRA dans mistral.rs | **Renommer ou abandonner** |
| `candle-quant` | libre | `mistralrs-quant` 179 k dl — GPTQ, AWQ, HQQ, FP8, BNB, ISQ | Réévaluer |
| `candle-attn` | libre | `candle-vllm` 713 ⭐ + `mistralrs-paged-attn` 73 k dl ; `EricLBuehler/candle_graphs` pour les CUDA graphs | Réévaluer |
| `candle-models` | libre | `candle-transformers` upstream, 3 M dl — c'est littéralement ce crate | Garder en interne seulement |
| `candle-decoding` | libre | `llguidance` 988 k dl (guidance-ai) + `outlines-core` + `EricLBuehler/candle-sampling` | Réévaluer |

**Le point dur : `candle-lora`.** Le nom est libre sur crates.io uniquement parce
qu'Eric Buehler n'a jamais publié le sien — son `Cargo.toml` déclare bien
`name = "candle-lora"`. Publier sous ce nom reviendrait à prendre celui d'un projet
connu et vivant. Buehler n'est pas un inconnu de l'écosystème : mistral.rs (7 600 ⭐)
est à lui, et **candle lui-même dépend de sa crate `float8`** (`candle-core/src/dtype.rs`
fait `use float8::F8E4M3`). Juste après un retour sur la précipitation, c'est le
signal à ne pas envoyer.

**Ce qui reste défendable.** NVFP4 est le seul sujet où personne n'est installé :
mistral.rs annonce GGUF/GPTQ/AWQ/HQQ/FP8/BNB mais pas NVFP4, et `float4` couvre
MXFP4, un format voisin mais distinct. C'est là que le travail du fork a une valeur
propre plutôt qu'une redite.

Pour les quatre « réévaluer », la question honnête n'est pas « est-ce que je peux le
réécrire » mais « qu'est-ce que Tachyon a besoin de faire que `mistralrs-quant`,
`candle-vllm` et `llguidance` ne font pas ». Si la réponse est courte, la bonne
réponse est une couche fine par-dessus, pas une pile de plus.

**Fait le 17 août 2026 : `candle-nvfp4` initialisé et peuplé.**
[astorise/candle-nvfp4](https://github.com/astorise/candle-nvfp4) porte le code du
fork depuis l'état de #3883 — deux crates (`candle-nvfp4` pour le chargement de
checkpoint + dispatch, `candle-nvfp4-kernels` pour le noyau CUDA optionnel), toutes
deux contre des versions crates.io publiées (`candle-core`/`candle-nn` 0.11,
`float8` 0.7), sans dépendance de chemin vers le fork. `cargo test --workspace`
passe (12/12) sur les fonctionnalités par défaut ; `nvfp4-cuda` n'a pas pu être
testé ici faute de toolchain CUDA dans ce bac à sable. Les quatre autres crates
« réévaluer » et `candle-lora` (à renommer) restent à traiter.

### 3.6 #3838 — l'exception, et ce qu'il faut y corriger

Gardée ouverte malgré ses 3 503 lignes, et c'est justifié : c'est la seule PR du lot
qui ait une **validation indépendante**. `@oetiker` l'a testée contre un checkpoint
Qwen3.6-35B réel et y a trouvé trois bugs numériques :

- le loader appliquait `-exp(A_log)` deux fois à `ssm_a` ;
- la table d'angles RoPE perdait en précision aux longues positions (position 5304
  décalée de 0,812 rad) ;
- le tiling de broadcast Q/K répétait en interleaved au lieu de cyclique.

Les trois correctifs sont dans la PR. **Aucun n'est attribué** : les cinq commits sont
signés `Sébastien ASTORI` seul, sans trailer `Co-authored-by`. À corriger — c'est dû, et
c'est aussi l'argument le plus fort de la PR. Une contribution de modèle validée par un
tiers ne se lit pas du tout comme un dépôt de 3 500 lignes en solo.

Trailer à utiliser (forme noreply publique, pas besoin de son adresse privée) :

```
Co-authored-by: oetiker <429279+oetiker@users.noreply.github.com>
```

**État au 17 août :** le commentaire proposant le crédit est posté sur la PR. Les cinq
commits ne portent **toujours pas** le trailer et oetiker n'a pas encore répondu. Tant
que c'est le cas, dire « offered him co-authorship » et non « credited him » — le
mainteneur peut vérifier l'historique.

**Ne pas découper la PR maintenant.** Le découpage naturel existe (le chemin quantifié —
`quantized_qwen3_5.rs` 768 L + son exemple 322 L — face au modèle dense, 2 373 L), et
#3865 trace déjà le MoE quantifié séparément. Mais oetiker a testé **les deux** chemins :
découper casserait sa base de test en cours. La collaboration vaut plus que le compte de
lignes. Le découpage se fera si le mainteneur le demande.

La description actuelle (~450–500 caractères, sections « Changes » / « Tests ») **n'est
pas** un essai généré : elle est déjà courte. Il lui manque seulement le fait le plus
vendeur, actuellement enterré dans le fil des commentaires. Textes en §5.5.

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

> You're right, and thanks for being direct.
>
> I've closed 39 of my 41 open PRs. Sixteen carried kernel crates that only exist
> in my fork — that work is moving to separate crates outside candle. Most of the
> rest were overlapping series where I'd opened a new PR instead of updating the
> branch.
>
> Two are left. #3894 is a one-line conv2d fix. #3838 I kept because @oetiker has
> been testing it against a real Qwen3.6-35B checkpoint on Metal and found three
> numerical bugs; his fixes are in and I've offered him co-authorship on the
> commits carrying them. Nothing new from me until both are resolved.
>
> Separately, in case it's useful: `candle-core` looks up `ucopy_f8e4m3` while
> `unary.cu` defines `ucopy_f8_e4m3` (`DType::as_str()` returns `"f8e4m3"`), so
> every F8E4M3 CUDA kernel is unreachable. Happy to send that as a standalone
> rename.

### 5.2 Commentaire de fermeture (motif A)

> Closing — this carries kernel crates that don't exist upstream. Moving it to a
> separate crate.

### 5.3 Commentaires de fermeture (motif B) — un par série

Ne citer aucune PR sœur : elles ferment toutes. La seule référence utile est #3894,
qui reste ouverte.

**#3821, #3838, #3869** — série Qwen 3.5

> Closing — I opened four overlapping PRs for Qwen 3.5 support instead of iterating
> on one branch. Closing all four; this work is moving to a separate crate.

**#3729** — série CUDA graph

> Closing — I opened three overlapping PRs for CUDA graph capture instead of
> iterating on one branch. Closing all three; this work is moving to a separate crate.

**#3789, #3811** — série IncrementalDecoder

> Closing — these two PRs are divergent branches of the same change. Closing both;
> this work is moving to a separate crate.

**#3895** — PR fourre-tout

> Closing — this bundles five unrelated changes (cudarc bump, error messages, f8e4m3
> rename, conv1d and conv2d fixes). I'm sending them separately; #3894 is the first.

### 5.4 Commentaire de fermeture (motif C)

> Closing — too broad for now. I'll revisit as a smaller, isolated change if
> there's interest.

### 5.4bis Commentaire de fermeture (motif D)

> Closing to trim my open queue. I still think this one stands on its own — I'll
> resubmit it later, on its own.

### 5.5 #3838 — créditer oetiker et remonter la validation

**Réponse dans le fil**

> @oetiker your three fixes are in, but I squashed them into my own commits and you
> ended up with no attribution — sorry about that. I'll add
> `Co-authored-by: oetiker <429279+oetiker@users.noreply.github.com>` to the commits
> carrying them unless you'd rather I use a different address.

**Description mise à jour** (remplace l'actuelle)

> Adds Qwen 3.5 support: sparse-MoE feed-forward blocks and partial RoPE. Fixes #3837.
>
> Tested against a released Qwen3.6-35B checkpoint by @oetiker, who found and fixed
> three numerical bugs now folded in: the loader applied `-exp(A_log)` twice to
> `ssm_a`, the RoPE angle table lost precision at long positions (position 5304 was
> off by 0.812 rad), and Q/K broadcast tiling repeated interleaved instead of cyclic.
>
> Quantized MoE is tracked separately in #3865.

### 5.6 Descriptions réécrites

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
