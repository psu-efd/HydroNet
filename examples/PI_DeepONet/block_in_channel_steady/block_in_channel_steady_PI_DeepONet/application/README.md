## This is the model application directory. 

Once a model is trained, it can be applied to the evaluation cases. What is needed for model application:
- The trained model checkpoint file
- The evaluation parameters file: These parameters (dimensional) are the branch inputs of the DeepONet.
- The evaluation vtk file (unstructured grid format): The evaluation is on the cell centers of the unstructured grid mesh.

The model predictions are saved in the result vtk files.

## Notes
- The evaluation parameters must be compatible with the branch inputs of the DeepONet.
- The evaluation parameters in the file are assumed to be dimensional. They will be normalized using the specified normalization method (and should be the same as the normalization method used during training).