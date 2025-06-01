import { useState, useEffect } from 'react';
import { ScaleIcon, ArrowTopRightOnSquareIcon, ChartBarIcon, TableCellsIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';
import { getLeaderboardByTaskId } from '../services/federatedService';
import { useParams } from 'react-router-dom';
import { Link } from 'react-router-dom';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';

const Leaderboard = () => {
  const { task_id } = useParams();
  const [isLoading, setIsLoading] = useState(true);
  const [leaderboardData, setLeaderboardData] = useState(null);
  const [error, setError] = useState(null);
  const { api } = useAuth();
  const [viewMode, setViewMode] = useState('table'); // 'table' or 'chart'

  useEffect(() => {
    const fetchLeaderboardHistory = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await getLeaderboardByTaskId(api, task_id);
        setLeaderboardData(response.data);
      } catch (err) {
        console.error("Error fetching leaderboard:", err);
        setError("Failed to fetch leaderboard data. Please try again.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchLeaderboardHistory();
  }, [api, task_id]);


  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;
  
    const data = payload[0].payload;
    if (!data) return null;
  
    return (
      <div className="bg-white p-4 border border-gray-200 rounded shadow-lg">
        <p className="font-bold">Session #{data.session_id || 'N/A'}</p>
        <p>Model: {data.model_name || 'N/A'}</p>
        <p>Client: {data.admin_username || 'N/A'}</p>
        <p>Value: {data.metric_value ? data.metric_value.toFixed(3) : 'N/A'}</p>
        <p>Date: {data.created_at ? new Date(data.created_at).toLocaleString() : 'N/A'}</p>
      </div>
    );
  };
  const processData = (sessions) => {
    if (!sessions) return [];

    // Sort sessions by date first (in ascending order)
    const sortedSessions = [...sessions].sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
    
    return sortedSessions.map(session => {
      const date = new Date(session.created_at);
      const month = date.getMonth() + 1; // Months are 0-indexed
      const year = date.getFullYear();
      return {
        ...session,
        date: new Date(session.created_at),
        monthYear: `${String(month).padStart(2, '0')}/${String(year).slice(-2)}`,
        formattedDate: date.toLocaleString('en-IN', {
          day: '2-digit',
          month: 'short',
          year: '2-digit',
          timeZone: 'Asia/Kolkata' // <-- Important for Indian time
        })
      };
    });
  };
  const processedData = processData(leaderboardData?.sessions || []);

  return (
    <div className="bg-white shadow-lg rounded-lg border border-gray-200 overflow-hidden">
      <div className="p-4 overflow-y-auto">
        {isLoading ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400"></div>
          </div>
        ) : error ? (
          <div className="rounded-md bg-red-50 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <XMarkIcon className="h-5 w-5 text-red-400" aria-hidden="true" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">{error}</h3>
              </div>
            </div>
          </div>
        ) : leaderboardData ? (
          <>
            <div className="p-4 border-b border-gray-200">
  <h3 className="font-medium text-gray-900 text-center">{leaderboardData.task_name} Leaderboard</h3>
  <div className="flex justify-center mt-2 space-x-2">
    <button
      onClick={() => setViewMode('table')}
      className={`p-2 rounded-md ${viewMode === 'table' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}`}
    >
      <TableCellsIcon className="h-5 w-5" />
    </button>
    <button
      onClick={() => setViewMode('chart')}
      className={`p-2 rounded-md ${viewMode === 'chart' ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}`}
    >
      <ChartBarIcon className="h-5 w-5" />
    </button>
  </div>
</div>


            {viewMode === 'table' ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Organization</th>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Client</th>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                      <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {/* Benchmark Row */}
                    <tr className="bg-green-50">
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-green-800">—</td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-green-800">Benchmark</td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-green-800">—</td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-green-800">—</td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-mono font-medium text-green-800">
                        {leaderboardData.benchmark || 'N/A'}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-green-800">
                        {new Date(leaderboardData.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-green-800">
                        {leaderboardData.metric === 'mae' || leaderboardData.metric === 'mse' 
                          ? 'Lower is better' 
                          : 'Higher is better'}
                      </td>
                    </tr>

                    {/* Session Rows */}
                    {leaderboardData.sessions
                      .sort((a, b) => {
                        if (leaderboardData.metric === 'mae' || leaderboardData.metric === 'mse') {
                          return a.metric_value - b.metric_value;
                        }
                        return b.metric_value - a.metric_value;
                      })
                      .map((session, index) => {
                        const isBetter = leaderboardData.benchmark && 
                          ((leaderboardData.metric === 'mae' || leaderboardData.metric === 'mse') 
                            ? session.metric_value <= leaderboardData.benchmark
                            : session.metric_value >= leaderboardData.benchmark);
                        
                        return (
                          <tr 
                            key={session.session_id} 
                            className={`hover:bg-gray-50 ${isBetter ? 'bg-green-50' : ''}`}
                          >
                            <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                              {index + 1}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                              {session.organisation_name}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                              {session.admin_username}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                              {session.model_name}
                            </td>
                            <td className={`px-4 py-3 whitespace-nowrap text-sm font-mono font-medium ${
                              isBetter ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {session.metric_value.toFixed(3)}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                              {new Date(session.created_at).toLocaleDateString()}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                              <Link
                                to={`/trainings/${session.session_id}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center px-2 py-1 border border-gray-300 rounded-md shadow-sm text-xs font-medium text-gray-700 bg-white hover:bg-gray-50"
                              >
                                <ArrowTopRightOnSquareIcon className="h-3 w-3 mr-1" />
                                View
                              </Link>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-4">
  <ResponsiveContainer width="100%" height={500}>
    <ScatterChart margin={{ top: 20, right: 100, bottom: 40, left: 100 }}>
      <CartesianGrid strokeDasharray="3 3" />
      
      <XAxis 
  dataKey="formattedDate" // use preprocessed date string
  type="category"         // <- key change: use 'category' for discrete dates
  name="Date"
  tick={{ fontSize: 12 }}
  label={{ 
    value: 'Date (DD/MM/YY)', 
    position: 'insideBottomRight', 
    offset: -5 
  }}
/>
      
      <YAxis 
        dataKey="metric_value" 
        name="Value"
        label={{ 
          value: leaderboardData.metric?.toUpperCase(), 
          angle: -90, 
          position: 'insideLeft' 
        }}
      />
      
      <ZAxis 
        dataKey="total_rounds" 
        range={[50, 300]} 
        name="Rounds" 
      />
      
      <Tooltip content={<CustomTooltip />} />
      <Legend />
      
      {leaderboardData?.benchmark && (
        <ReferenceLine 
          y={leaderboardData.benchmark} 
          stroke="green" 
          label={{
            value: 'Benchmark', 
            position: 'right',
            fill: 'green'
          }}
          strokeDasharray="5 5"
        />
      )}
      
      <Scatter name="Training Sessions" data={processedData}>
        {processedData.map((entry, index) => (
          <Cell 
            key={`cell-${index}`} 
            fill={entry.meets_benchmark ? '#82ca9d' : '#ff6b6b'} 
          />
        ))}
      </Scatter>
    </ScatterChart>
  </ResponsiveContainer>

  <div className="flex justify-center mt-2 space-x-4">
    <div className="flex items-center">
      <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
      <span className="text-xs">Better than benchmark</span>
    </div>
    <div className="flex items-center">
      <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
      <span className="text-xs">Worse than benchmark</span>
    </div>
  </div>
</div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
};

export default Leaderboard;