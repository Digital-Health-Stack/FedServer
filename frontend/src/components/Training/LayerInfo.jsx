const LayerInfo = ({ layer, index }) => (
  <div className="p-4 border rounded-lg shadow-sm hover:shadow-md transition-shadow mt-4">
    <h4 className="font-bold text-indigo-600 mb-2">
      {index + 1}.{" "}
      {layer.layer_type.charAt(0).toUpperCase() +
        layer.layer_type.slice(1).replace("_", " ")}
    </h4>
    <ul className="space-y-2">
      {/* Convolution Layer */}
      {layer.layer_type === "convolution" && (
        <>
          <li className="flex justify-between">
            <span className="text-gray-600">Filters:</span>
            <span className="font-medium">{layer.filters}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Kernel Size:</span>
            <span className="font-medium">{layer.kernel_size}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Stride:</span>
            <span className="font-medium">{layer.stride}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Padding:</span>
            <span className="font-medium">{layer.padding || "valid"}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Activation:</span>
            <span className="font-medium">{layer.activation_function}</span>
          </li>
        </>
      )}

      {/* Pooling Layer */}
      {layer.layer_type === "pooling" && (
        <>
          <li className="flex justify-between">
            <span className="text-gray-600">Type:</span>
            <span className="font-medium">{layer.pooling_type}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Pool Size:</span>
            <span className="font-medium">{layer.pool_size}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Stride:</span>
            <span className="font-medium">{layer.stride}</span>
          </li>
        </>
      )}

      {/* Dense Layer */}
      {layer.layer_type === "dense" && (
        <>
          <li className="flex justify-between">
            <span className="text-gray-600">Units:</span>
            <span className="font-medium">{layer.num_nodes}</span>
          </li>
          <li className="flex justify-between">
            <span className="text-gray-600">Activation:</span>
            <span className="font-medium">{layer.activation_function}</span>
          </li>
          {layer.regularizer && (
            <li className="flex justify-between">
              <span className="text-gray-600">Regularizer:</span>
              <span className="font-medium">
                {layer.regularizer.type} ({layer.regularizer.factor})
              </span>
            </li>
          )}
        </>
      )}

      {/* Batch Normalization */}
      {layer.layer_type === "batch_norm" && (
        <li className="text-gray-500 italic">
          Normalizes the activations of the previous layer
        </li>
      )}

      {/* Dropout */}
      {layer.layer_type === "dropout" && (
        <li className="flex justify-between">
          <span className="text-gray-600">Rate:</span>
          <span className="font-medium">{layer.rate}</span>
        </li>
      )}

      {/* Flatten */}
      {layer.layer_type === "flatten" && (
        <li className="text-gray-500 italic">
          Flattens the input without affecting batch size
        </li>
      )}

      {/* Reshape */}
      {layer.layer_type === "reshape" && (
        <li className="flex justify-between">
          <span className="text-gray-600">Target Shape:</span>
          <span className="font-medium">{layer.target_shape}</span>
        </li>
      )}
    </ul>
  </div>
);

export default LayerInfo;
