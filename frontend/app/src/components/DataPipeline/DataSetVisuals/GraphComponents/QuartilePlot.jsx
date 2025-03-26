import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { InformationCircleIcon } from "@heroicons/react/24/solid";

const QuartilePlot = ({ quartiles }) => {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(800);
  const height = 140;
  const margin = { top: 30, right: 20, bottom: 40, left: 20 };

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      if (entries[0].contentRect.width) {
        setWidth(entries[0].contentRect.width - margin.left - margin.right);
      }
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  const { min = 0, Q1 = 0, median = 0, Q3 = 0, max = 0 } = quartiles || {};
  const scale = d3.scaleLinear().domain([min, max]).range([0, width]).nice();

  const isValidData = [min, Q1, median, Q3, max].every(Number.isFinite);
  const iqr = Q3 - Q1;
  const showBox = isValidData && iqr > 0;

  return (
    <div
      ref={containerRef}
      className="w-full p-4 bg-white rounded-xl shadow-sm border border-gray-100 relative"
    >
      {isValidData ? (
        <svg width="100%" height={height} className="overflow-visible">
          <g transform={`translate(${margin.left},${height / 2})`}>
            {/* Whiskers */}
            <path
              d={`M${scale(min)} 0 H${scale(max)}`}
              stroke="#64748b"
              strokeWidth="1.5"
              strokeDasharray={showBox ? "4 2" : "0"}
            />

            {/* Box */}
            {showBox && (
              <g>
                <rect
                  x={scale(Q1)}
                  y={-height / 4}
                  width={scale(Q3) - scale(Q1)}
                  height={height / 2}
                  fill="#e0f2fe"
                  stroke="#0ea5e9"
                  strokeWidth="1.5"
                  rx="3"
                />
                {/* Quartile Labels */}
                <g transform={`translate(${scale(Q1)},${height / 4 + 20})`}>
                  <text
                    textAnchor="middle"
                    fill="#475569"
                    fontSize="10"
                    className="font-medium"
                  >
                    Q1: {Number(Q1).toFixed(2)}
                  </text>
                </g>
                <g transform={`translate(${scale(Q3)},${height / 4 + 20})`}>
                  <text
                    textAnchor="middle"
                    fill="#475569"
                    fontSize="10"
                    className="font-medium"
                  >
                    Q3: {Number(Q3).toFixed(2)}
                  </text>
                </g>
              </g>
            )}

            {/* Median Line */}
            <path
              d={`M${scale(median)} ${-height / 4} V${height / 4}`}
              stroke="#0369a1"
              strokeWidth="2"
            />

            {/* End Points */}
            <circle cx={scale(min)} r="4" fill="#64748b" />
            <circle cx={scale(max)} r="4" fill="#64748b" />

            {/* Axis Labels */}
            <g transform={`translate(${scale(min)},20)`}>
              <text
                textAnchor="middle"
                fill="#475569"
                fontSize="10"
                className="font-semibold"
              >
                {Number(min).toLocaleString(undefined, {
                  maximumFractionDigits: 2,
                })}
              </text>
            </g>
            <g transform={`translate(${scale(max)},20)`}>
              <text
                textAnchor="middle"
                fill="#475569"
                fontSize="10"
                className="font-semibold"
              >
                {Number(max).toLocaleString(undefined, {
                  maximumFractionDigits: 2,
                })}
              </text>
            </g>
            <g transform={`translate(${scale(median)},-${height / 4 + 10})`}>
              <text
                textAnchor="middle"
                fill="#0369a1"
                fontSize="10"
                className="font-semibold"
              >
                Median: {Number(median).toFixed(2)}
              </text>
            </g>
          </g>
        </svg>
      ) : (
        <div className="h-full flex items-center justify-center text-gray-400 text-sm">
          <InformationCircleIcon className="w-5 h-5 mr-2" />
          No valid quartile data available
        </div>
      )}
    </div>
  );
};

export default QuartilePlot;
