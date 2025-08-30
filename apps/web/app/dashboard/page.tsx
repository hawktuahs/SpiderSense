'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, BarChart3, AlertTriangle, Download, Settings } from 'lucide-react'
import toast from 'react-hot-toast'

interface UploadedFile {
  file: File
  preview: string
}

export default function DashboardPage() {
  const [step, setStep] = useState(1)
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null)
  const [seriesId, setSeriesId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setUploadedFile({
        file,
        preview: URL.createObjectURL(file)
      })
      setStep(2)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    maxFiles: 1
  })

  const handleUpload = async () => {
    if (!uploadedFile) return

    setIsLoading(true)
    const formData = new FormData()
    formData.append('file', uploadedFile.file)

    try {
      const response = await fetch('/api/ingest', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const result = await response.json()
      setSeriesId(result.series_id)
      setStep(3)
      toast.success('Data uploaded successfully!')
    } catch (error) {
      toast.error('Upload failed. Please try again.')
      console.error('Upload error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadDemoData = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/demo/load')
      if (!response.ok) {
        throw new Error('Demo load failed')
      }
      
      const result = await response.json()
      if (result.series && result.series.length > 0) {
        setSeriesId(result.series[0].series_id)
        setStep(3)
        toast.success('Demo data loaded!')
      }
    } catch (error) {
      toast.error('Failed to load demo data')
      console.error('Demo load error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-orange-400 via-white to-green-500 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-gray-800" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">SmartSense Dashboard</h1>
                <p className="text-xs text-gray-500">Energy Load Forecasting & Anomaly Detection</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button className="btn-outline btn-sm">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Steps */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            {[
              { step: 1, title: 'Upload Data', icon: Upload },
              { step: 2, title: 'Configure', icon: Settings },
              { step: 3, title: 'Forecast', icon: BarChart3 },
              { step: 4, title: 'Anomalies', icon: AlertTriangle },
              { step: 5, title: 'Export', icon: Download }
            ].map((item, index) => (
              <div key={item.step} className="flex items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  step >= item.step 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-gray-200 text-gray-500'
                }`}>
                  <item.icon className="w-4 h-4" />
                </div>
                <span className={`ml-2 text-sm font-medium ${
                  step >= item.step ? 'text-gray-900' : 'text-gray-500'
                }`}>
                  {item.title}
                </span>
                {index < 4 && (
                  <div className={`w-12 h-0.5 mx-4 ${
                    step > item.step ? 'bg-primary-600' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {step === 1 && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Energy Data</h2>
              <p className="text-gray-600">
                Upload your CSV file with energy consumption data or try our demo dataset
              </p>
            </div>

            {/* Upload Area */}
            <div className="card p-8 mb-6">
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive 
                    ? 'border-primary-500 bg-primary-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-900 mb-2">
                  {isDragActive ? 'Drop your CSV file here' : 'Drag & drop your CSV file here'}
                </p>
                <p className="text-gray-500 mb-4">or click to browse files</p>
                <div className="text-sm text-gray-400">
                  Supported format: CSV with timestamp and value columns
                </div>
              </div>

              {uploadedFile && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <FileText className="w-5 h-5 text-gray-500" />
                      <div>
                        <p className="font-medium text-gray-900">{uploadedFile.file.name}</p>
                        <p className="text-sm text-gray-500">
                          {(uploadedFile.file.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={handleUpload}
                      disabled={isLoading}
                      className="btn-primary"
                    >
                      {isLoading ? 'Uploading...' : 'Process File'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Demo Option */}
            <div className="text-center">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-300" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-2 bg-gray-50 text-gray-500">or</span>
                </div>
              </div>
              
              <div className="mt-6">
                <button
                  onClick={loadDemoData}
                  disabled={isLoading}
                  className="btn-outline btn-lg"
                >
                  <BarChart3 className="w-5 h-5 mr-2" />
                  {isLoading ? 'Loading...' : 'Try Demo Dataset'}
                </button>
                <p className="text-sm text-gray-500 mt-2">
                  Sample energy data from Indian commercial building
                </p>
              </div>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Configure Analysis</h2>
              <p className="text-gray-600">
                Set up your forecasting and anomaly detection parameters
              </p>
            </div>

            <div className="card p-6 space-y-6">
              <div>
                <label className="label mb-2">Forecast Horizon</label>
                <select className="input">
                  <option value="24">24 hours</option>
                  <option value="48">48 hours</option>
                  <option value="168">1 week</option>
                </select>
              </div>

              <div>
                <label className="label mb-2">Forecasting Model</label>
                <select className="input">
                  <option value="arima">Auto-ARIMA</option>
                  <option value="ets">Exponential Smoothing</option>
                  <option value="naive_seasonal">Seasonal Naive</option>
                </select>
              </div>

              <div>
                <label className="label mb-2">City (for weather data)</label>
                <select className="input">
                  <option value="mumbai">Mumbai</option>
                  <option value="delhi">Delhi</option>
                  <option value="bangalore">Bangalore</option>
                  <option value="chennai">Chennai</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                <input type="checkbox" id="weather" className="rounded" defaultChecked />
                <label htmlFor="weather" className="text-sm text-gray-700">
                  Include weather forecasts
                </label>
              </div>

              <div className="flex items-center space-x-2">
                <input type="checkbox" id="anomalies" className="rounded" defaultChecked />
                <label htmlFor="anomalies" className="text-sm text-gray-700">
                  Enable anomaly detection
                </label>
              </div>

              <button
                onClick={() => setStep(3)}
                className="btn-primary w-full"
              >
                Generate Forecast
              </button>
            </div>
          </div>
        )}

        {step === 3 && seriesId && (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Energy Load Forecast</h2>
              <p className="text-gray-600">
                48-hour forecast with 90% prediction intervals
              </p>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Main Chart */}
              <div className="lg:col-span-2">
                <div className="card p-6">
                  <div className="h-80 bg-gray-50 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-500">Interactive forecast chart will appear here</p>
                      <p className="text-sm text-gray-400 mt-2">
                        Showing historical data + 48h forecast with confidence bands
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-sm">
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-blue-500 rounded mr-2"></div>
                        Historical
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-green-500 rounded mr-2"></div>
                        Forecast
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-gray-300 rounded mr-2"></div>
                        Confidence Interval
                      </div>
                    </div>
                    <button
                      onClick={() => setStep(4)}
                      className="btn-primary btn-sm"
                    >
                      View Anomalies
                    </button>
                  </div>
                </div>
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Forecast Summary */}
                <div className="card p-4">
                  <h3 className="font-semibold mb-3">Forecast Summary</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Model:</span>
                      <span className="font-medium">Auto-ARIMA</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Horizon:</span>
                      <span className="font-medium">48 hours</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">MAPE:</span>
                      <span className="font-medium text-green-600">8.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Next 24h Avg:</span>
                      <span className="font-medium">125.4 kWh</span>
                    </div>
                  </div>
                </div>

                {/* Weather Impact */}
                <div className="card p-4">
                  <h3 className="font-semibold mb-3">Weather Impact</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Temperature:</span>
                      <span className="font-medium">28°C ↑</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Humidity:</span>
                      <span className="font-medium">65% ↓</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Impact:</span>
                      <span className="font-medium text-orange-600">+12% load</span>
                    </div>
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="card p-4">
                  <h3 className="font-semibold mb-3">Quick Actions</h3>
                  <div className="space-y-2">
                    <button className="btn-outline w-full text-sm">
                      <Download className="w-4 h-4 mr-2" />
                      Export CSV
                    </button>
                    <button className="btn-outline w-full text-sm">
                      <BarChart3 className="w-4 h-4 mr-2" />
                      Run Backtest
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 4 && (
          <div>
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Anomaly Detection</h2>
              <p className="text-gray-600">
                Real-time anomaly alerts and pattern analysis
              </p>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Anomaly Chart */}
              <div className="lg:col-span-2">
                <div className="card p-6">
                  <div className="h-80 bg-gray-50 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <AlertTriangle className="w-16 h-16 text-orange-400 mx-auto mb-4" />
                      <p className="text-gray-500">Anomaly detection visualization</p>
                      <p className="text-sm text-gray-400 mt-2">
                        Showing detected anomalies with severity levels
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Anomaly List */}
              <div>
                <div className="card p-4">
                  <h3 className="font-semibold mb-3">Recent Anomalies</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <span className="badge-danger">High Spike</span>
                        <span className="text-xs text-gray-500">2h ago</span>
                      </div>
                      <p className="text-sm text-gray-700">
                        +3.2σ deviation at 14:00
                      </p>
                    </div>
                    
                    <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <span className="badge-warning">Medium Drop</span>
                        <span className="text-xs text-gray-500">6h ago</span>
                      </div>
                      <p className="text-sm text-gray-700">
                        -2.1σ deviation at 10:00
                      </p>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => setStep(5)}
                    className="btn-primary w-full mt-4"
                  >
                    Export Results
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 5 && (
          <div className="max-w-2xl mx-auto text-center">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Export Results</h2>
              <p className="text-gray-600">
                Download your forecasts, anomalies, and analysis reports
              </p>
            </div>

            <div className="card p-6 space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <button className="btn-outline p-4 h-auto">
                  <Download className="w-6 h-6 mx-auto mb-2" />
                  <div className="text-sm font-medium">Forecast CSV</div>
                  <div className="text-xs text-gray-500">Timestamps & predictions</div>
                </button>
                
                <button className="btn-outline p-4 h-auto">
                  <Download className="w-6 h-6 mx-auto mb-2" />
                  <div className="text-sm font-medium">Anomaly Report</div>
                  <div className="text-xs text-gray-500">Detected anomalies & scores</div>
                </button>
                
                <button className="btn-outline p-4 h-auto">
                  <Download className="w-6 h-6 mx-auto mb-2" />
                  <div className="text-sm font-medium">Model Metrics</div>
                  <div className="text-xs text-gray-500">Performance statistics</div>
                </button>
                
                <button className="btn-outline p-4 h-auto">
                  <Download className="w-6 h-6 mx-auto mb-2" />
                  <div className="text-sm font-medium">Visualization</div>
                  <div className="text-xs text-gray-500">Charts & plots (PNG)</div>
                </button>
              </div>
              
              <div className="pt-4 border-t">
                <button
                  onClick={() => setStep(1)}
                  className="btn-primary w-full"
                >
                  Start New Analysis
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
