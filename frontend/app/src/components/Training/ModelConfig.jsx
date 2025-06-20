import { InformationCircleIcon } from "@heroicons/react/24/outline";
import InfoItem from "./InfoItem";
import LayerInfo from "./LayerInfo";

const ModelConfig = ({ data }) => (
  <div className="bg-white p-5 rounded-xl border border-gray-100">
    <h3 className="text-xl font-semibold mb-4 text-gray-800 border-b pb-2 flex items-center">
      <InformationCircleIcon className="h-5 w-5 text-indigo-700 mr-2" />
      Model Configuration
    </h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-gray-700">
      <InfoItem label="Input Shape" value={data?.model_info?.input_shape} />
      <InfoItem label="Loss Function" value={data?.model_info?.loss} />
      <InfoItem
        label="Optimizer"
        value={`${data?.model_info?.optimizer?.type} (lr: ${data?.model_info?.optimizer?.learning_rate})`}
      />
      <InfoItem
        label="Output Nodes"
        value={data?.model_info?.output_layer?.num_nodes}
      />
      <InfoItem
        label="Output Activation"
        value={data?.model_info?.output_layer?.activation_function}
      />
    </div>

    {/* Metrics Section */}
    <div className="mt-4">
      <h4 className="text-sm font-medium text-gray-700 mb-2">Metrics</h4>
      <div className="flex flex-wrap gap-2">
        {Object.entries(data?.model_info?.metrics || {}).map(
          ([metric, enabled]) =>
            enabled && (
              <span
                key={metric}
                className="px-2 py-1 bg-blue-50 text-blue-700 rounded-full text-xs"
              >
                {metric.toUpperCase()}
              </span>
            ),
        )}
      </div>
    </div>

    {/* Test Metrics Section */}
    {data?.model_info?.test_metrics?.length > 0 && (
      <div className="mt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Test Metrics</h4>
        <div className="flex flex-wrap gap-2">
          {data.model_info.test_metrics.map((metric) => (
            <span
              key={metric}
              className="px-2 py-1 bg-green-50 text-green-700 rounded-full text-xs"
            >
              {metric.toUpperCase()}
            </span>
          ))}
        </div>
      </div>
    )}

    {/* Model Architecture - Layers */}
    <div className="mt-6">
      <h4 className="text-lg font-semibold text-gray-700 mb-2">
        Model Architecture (Layers)
      </h4>
      {data?.model_info?.layers?.length > 0 ? (
        <div className="space-y-3">
          {data.model_info.layers.map((layer, index) => (
            <LayerInfo key={index} layer={layer} index={index} />
          ))}
        </div>
      ) : (
        <div className="text-gray-400">No layer information available</div>
      )}
    </div>
  </div>
);

export default ModelConfig;
