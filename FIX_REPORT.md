### Bug 1: Label Tensor Type Incorrect in Training
File: ml/data.py

Line: In ReviewDataset.__getitem__, label tensor returned as torch.float

Issue:
The model uses PyTorch's CrossEntropyLoss for classification. This loss function expects target labels as integer tensor (torch.long), not floating-point (torch.float). Returning them as float caused a runtime error:
expected scalar type Long but found Float

### Bug 2: Model Not Invoked in /predict Endpoint
File: app/endpoints/routes.py

Line: In the /predict endpoint handler

Issue:
The route was returning label, confidence = (None, None) and attempted to format/log the confidence as a float using confidence:.4f. This resulted in the error:
unsupported format string passed to NoneType.__format__
Worse, the model prediction was never actually called, so no inference took place.

updated:
```python
 clf.predict(request.text)
```


Notes:
i also updated train.py to use ```mps``` which happens to be Mac GPU to speed up the training. but it's dynamic will work for cuda and cpu cases automatically.



## recommendation 
Bug 3: Vector Dimension Mismatch with Qdrant

File: db/vector_store.py

Line: In encode_text or embedding function for Qdrant

Issue:
Sent vectors of size 769 to Qdrant  by mistakenly appending an extra value to the embedding, causing Qdrant to reject all inserts.

updated code to 

```python
return embedding.flatten()
```
for qudrant i have setup locally via docker to make quick use of it on the same port and everything that in mentioned in the constructor .