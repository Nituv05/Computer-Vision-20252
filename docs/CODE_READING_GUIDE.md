# Huong dan doc code nhanh

Muc tieu: nam duoc duong di cua mot lan train M2-CL tu command line den model,
loss, checkpoint va ket qua.

## 0. Doc theo thu tu nay

1. `README.md`
   - Doc phan dau va cac command train/evaluate.
   - Muc tieu: biet repo co nhung method nao va chay bang lenh nao.

2. `tools/train.py`
   - Bat dau tu `main()`.
   - Di theo cac moc: parse args -> load config -> build datasets -> build
     algorithm -> train loop -> save checkpoint -> evaluate test.
   - Chua can doc tung baseline.

3. `algorithms/baselines.py`
   - Doc `METHODS`, `AlgorithmConfig`, `build_algorithm`.
   - Neu chi can hieu M2-CL, nhay thang vao `M2Algorithm`.
   - Dong quan trong: `logits, embeddings = self.model(x)` va
     `loss = self.criterion(logits, y, embeddings)`.

4. `models/m2cl.py`
   - Doc `build_model`, `OfficialM2.__init__`, `_register_hooks`, `forward`.
   - Muc tieu: hieu model lay intermediate feature bang hook, dua qua
     extraction block, concat lai va classify.

5. `models/extraction_block.py`
   - Doc theo thu tu `ConcentrationPipeline` -> `MLP` -> `ExtractionBlock`.
   - Muc tieu: hieu moi feature map di qua 1x1 conv, dropout, max pool, MLP.

6. `losses/contrastive.py`
   - Doc `MultiLayerContrastiveLoss.forward()` truoc.
   - Sau do doc `LayerContrastiveLoss.forward()`.
   - Muc tieu: hieu CE la loss chinh, contrastive score chi la regularizer.

7. `data/*.py` va `configs/*.yaml`
   - Doc sau cung.
   - Muc tieu: biet domain/class layout va default hyperparameters.

## 1. Duong di ngan nhat cua M2-CL

Command:

```bash
python tools/train.py --dataset pacs --test_domain photo --method m2cl
```

Flow:

```text
tools/train.py
  -> load configs/pacs.yaml
  -> build_datasets(...)
  -> split_source_environments(...)
  -> build_algorithm(method="m2cl")
  -> M2Algorithm.update(batch)
  -> OfficialM2.forward(x)
  -> ExtractionBlock.forward(feature_map)
  -> MultiLayerContrastiveLoss.forward(logits, labels, energies)
  -> optimizer.step()
  -> save best checkpoint by source validation accuracy
  -> evaluate held-out target domain
```

## 2. Nam nhanh tung module

`tools/train.py` la entrypoint. File nay tra loi cau hoi: chay dataset nao, target
domain nao, method nao, hyperparameter nao, checkpoint luu o dau.

`algorithms/baselines.py` la lop dieu phoi. No bien model/loss thanh mot API
chung co `update()` va `predict()`. Neu khong can baseline, chi doc
`M2Algorithm`.

`models/m2cl.py` la kien truc M2/M2-CL. Diem kho la forward hook: ResNet chay
binh thuong, hook bat lai feature o cac layer trung gian, roi extraction block
xu ly chung.

`models/extraction_block.py` la noi bien feature map lon thanh vector gon hon.
Tu khoa can nho: channel compression, spatial dropout, max pooling, MLP,
concatenation.

`losses/contrastive.py` la objective. `MultiLayerContrastiveLoss` tinh
cross-entropy truoc. Neu method la `m2cl`, no tru them `alpha * custom_score`.
Custom score lon khi cac mau cung class co representation giong nhau.

## 3. Neu chi co 30 phut

1. Doc `README.md` 5 phut.
2. Doc `tools/train.py main()` 8 phut.
3. Doc `M2Algorithm` trong `algorithms/baselines.py` 5 phut.
4. Doc `OfficialM2.forward()` trong `models/m2cl.py` 7 phut.
5. Doc `MultiLayerContrastiveLoss.forward()` 5 phut.

Sau 30 phut nay, ban se hieu duoc backbone cua project. Cac baseline nhu RSC,
CORAL, MMD, SAGM co the doc sau.

## 4. Cau hoi tu kiem tra

- Target domain co duoc dung de chon checkpoint khong?
- `m2` khac `m2cl` o dau?
- `embeddings`/`energies` tu model duoc dung vao loss nhu the nao?
- Vi sao batch size lon quan trong voi contrastive loss?
- Vi sao ResNet-50 khong concat tat ca extraction outputs?

Tra loi duoc 5 cau nay la da nam duoc y chinh cua codebase.
