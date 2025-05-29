import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";

// const TRAINING_URL = process.env.REACT_APP_GET_TRAINING_WITH_BENCHMARK_ID;

const BenchmarkTraining = () => {
  const { benchmark_id } = useParams();
  const [training, setTraining] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTraining = async () => {
      const response = await getTrainingWithBenchmark(benchmark_id);
      setTraining(response.data);
      setLoading(false);
    };
    fetchTraining();
  }, [benchmark_id]);

  return (
    <div className="p-4">
      <div className="bg-white p-4 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">
          Training Details for Benchmark {benchmark_id}
        </h2>

        {loading ? (
          <p>Loading training details...</p>
        ) : training ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-gray-500">Model Name:</p>
                <p className="font-medium">{training.model_name}</p>
              </div>
              <div>
                <p className="text-gray-500">Training Duration:</p>
                <p className="font-medium">{training.duration} minutes</p>
              </div>
              <div>
                <p className="text-gray-500">Accuracy:</p>
                <p className="font-medium">{training.accuracy.toFixed(2)}%</p>
              </div>
              <div>
                <p className="text-gray-500">Training Date:</p>
                <p className="font-medium">
                  {new Date(training.training_date).toLocaleDateString()}
                </p>
              </div>
            </div>
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Parameters</h3>
              <pre className="bg-gray-50 p-3 rounded">
                {JSON.stringify(training.parameters, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <p className="text-gray-500">No training details found</p>
        )}
      </div>
    </div>
  );
};

export default BenchmarkTraining;
