'use client'

import { useState } from 'react'
import { Zap, BarChart3, AlertTriangle, Globe, Upload, TrendingUp } from 'lucide-react'
import Link from 'next/link'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-orange-400 via-white to-green-500 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-gray-800" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">SmartSense</h1>
                <p className="text-xs text-gray-500">Energy Intelligence for India</p>
              </div>
            </div>
            <nav className="hidden md:flex items-center space-x-6">
              <Link href="/demo" className="text-gray-600 hover:text-gray-900 transition-colors">
                Demo
              </Link>
              <Link href="/docs" className="text-gray-600 hover:text-gray-900 transition-colors">
                Docs
              </Link>
              <Link href="/about" className="text-gray-600 hover:text-gray-900 transition-colors">
                About
              </Link>
              <Link href="/dashboard" className="btn-primary">
                Get Started
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <div className="inline-flex items-center px-4 py-2 bg-primary-100 text-primary-800 rounded-full text-sm font-medium mb-6">
              <Globe className="w-4 h-4 mr-2" />
              🇮🇳 Built for India's Smart Cities Mission
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6 text-balance">
              Scalable Energy Load
              <span className="bg-gradient-to-r from-orange-500 via-yellow-500 to-green-500 bg-clip-text text-transparent">
                {" "}Forecasting{" "}
              </span>
              & Anomaly Detection
            </h1>
            
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto text-balance">
              Open-source energy intelligence platform empowering municipalities, schools, 
              hospitals, and SMEs across India with 20-30% improved forecasting accuracy 
              and real-time anomaly alerts.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link href="/dashboard" className="btn-primary btn-lg">
                <Upload className="w-5 h-5 mr-2" />
                Upload Energy Data
              </Link>
              <Link href="/demo" className="btn-outline btn-lg">
                <BarChart3 className="w-5 h-5 mr-2" />
                View Live Demo
              </Link>
            </div>
            
            <div className="mt-8 text-sm text-gray-500">
              Free tier deployment • CPU-only • &lt;1GB memory • &lt;10s cold start
            </div>
          </div>
        </div>
      </section>

      {/* Impact Stats */}
      <section className="py-16 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary-600">20-30%</div>
              <div className="text-sm text-gray-600">Better Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">₹60,000</div>
              <div className="text-sm text-gray-600">Monthly Savings*</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600">160,000kg</div>
              <div className="text-sm text-gray-600">CO₂ Reduction*</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">&lt;10s</div>
              <div className="text-sm text-gray-600">Cold Start</div>
            </div>
          </div>
          <div className="text-center mt-4">
            <p className="text-xs text-gray-500">*Typical SME factory with 200,000 kWh monthly load</p>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Complete Energy Intelligence Platform
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              From data ingestion to actionable insights, everything you need 
              for smart energy management in one platform.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="card p-6 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
                <TrendingUp className="w-6 h-6 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Advanced Forecasting</h3>
              <p className="text-gray-600 mb-4">
                Multiple models including ARIMA, ETS, and NHITS-tiny with automatic 
                selection and prediction intervals.
              </p>
              <ul className="text-sm text-gray-500 space-y-1">
                <li>• 48-hour forecasts with 90% confidence intervals</li>
                <li>• Weather integration for improved accuracy</li>
                <li>• Multi-meter support for campus-wide analysis</li>
              </ul>
            </div>
            
            <div className="card p-6 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-danger-100 rounded-lg flex items-center justify-center mb-4">
                <AlertTriangle className="w-6 h-6 text-danger-600" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Real-time Anomaly Detection</h3>
              <p className="text-gray-600 mb-4">
                Robust statistical methods with seasonal awareness and 
                changepoint detection for comprehensive monitoring.
              </p>
              <ul className="text-sm text-gray-500 space-y-1">
                <li>• MAD-based robust z-score detection</li>
                <li>• EWMA for real-time streaming alerts</li>
                <li>• Regime shift identification</li>
              </ul>
            </div>
            
            <div className="card p-6 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-success-600" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Comprehensive Analytics</h3>
              <p className="text-gray-600 mb-4">
                Backtesting, model comparison, and detailed performance 
                metrics with exportable reports and visualizations.
              </p>
              <ul className="text-sm text-gray-500 space-y-1">
                <li>• Sliding window backtesting</li>
                <li>• Model leaderboards with MAPE/sMAPE</li>
                <li>• PNG/CSV export capabilities</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 bg-gradient-to-r from-blue-50 to-green-50 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Empowering India's Energy Transition
            </h2>
            <p className="text-lg text-gray-600">
              Supporting Smart Cities Mission and Net-Zero 2070 goals
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                title: "Municipal Offices",
                load: "50,000 kWh/month",
                savings: "₹15,000",
                co2: "40,000 kg",
                icon: "🏛️"
              },
              {
                title: "School Campus",
                load: "25,000 kWh/month", 
                savings: "₹7,500",
                co2: "20,000 kg",
                icon: "🏫"
              },
              {
                title: "Hospital Wing",
                load: "100,000 kWh/month",
                savings: "₹30,000", 
                co2: "80,000 kg",
                icon: "🏥"
              },
              {
                title: "SME Factory",
                load: "200,000 kWh/month",
                savings: "₹60,000",
                co2: "160,000 kg", 
                icon: "🏭"
              }
            ].map((useCase, index) => (
              <div key={index} className="card p-6 text-center">
                <div className="text-3xl mb-3">{useCase.icon}</div>
                <h3 className="font-semibold text-gray-900 mb-2">{useCase.title}</h3>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>{useCase.load}</div>
                  <div className="text-green-600 font-medium">Save {useCase.savings}/month</div>
                  <div className="text-blue-600 font-medium">Reduce {useCase.co2} CO₂</div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="text-center mt-8">
            <p className="text-sm text-gray-500">
              *Based on 20% efficiency improvement and ₹3/kWh average tariff
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Ready to Transform Your Energy Management?
          </h2>
          <p className="text-lg text-gray-600 mb-8">
            Start with our 10-minute quickstart or try the live demo with sample data.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/dashboard" className="btn-primary btn-lg">
              <Upload className="w-5 h-5 mr-2" />
              Upload Your Data
            </Link>
            <Link href="/demo" className="btn-outline btn-lg">
              <BarChart3 className="w-5 h-5 mr-2" />
              Try Live Demo
            </Link>
          </div>
          
          <div className="mt-8 flex items-center justify-center space-x-6 text-sm text-gray-500">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              MIT License
            </div>
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              Free Deployment
            </div>
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              No API Keys Required
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-6 h-6 bg-gradient-to-br from-orange-400 via-white to-green-500 rounded"></div>
                <span className="font-bold">SmartSense</span>
              </div>
              <p className="text-gray-400 text-sm">
                Open-source energy intelligence for India's Smart Cities Mission and Net-Zero goals.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3">Platform</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link></li>
                <li><Link href="/demo" className="hover:text-white transition-colors">Demo</Link></li>
                <li><Link href="/docs" className="hover:text-white transition-colors">Documentation</Link></li>
                <li><Link href="/api" className="hover:text-white transition-colors">API Reference</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3">Resources</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><Link href="/docs/quickstart" className="hover:text-white transition-colors">Quick Start</Link></li>
                <li><Link href="/docs/deploy" className="hover:text-white transition-colors">Deployment</Link></li>
                <li><Link href="/docs/impact" className="hover:text-white transition-colors">Impact Calculator</Link></li>
                <li><Link href="/examples" className="hover:text-white transition-colors">Examples</Link></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3">Community</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="https://github.com/smartsense" className="hover:text-white transition-colors">GitHub</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Discord</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Twitter</a></li>
                <li><a href="#" className="hover:text-white transition-colors">LinkedIn</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>© 2024 SmartSense. Open source under MIT License. Built for India's energy future.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
