"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

interface AnalysisResponse {
  company_info: {
    company: string;
    credit_grade: string;
    analysis_date: string;
    structured_score: string;
    final_fused_score: string;
    unstructured_score: string;
  };
  fusion_explanation: {
    explanation: string;
    fusion_method: string;
    market_regime: string;
    expert_agreement: number;
  };
  explainability_report?: string;
  historical_scores_summary: string;
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ ticker: string; name: string }[]>([]);
  const [ticker, setTicker] = useState("");
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [shouldSearch, setShouldSearch] = useState(true);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [error, setError] = useState("");

  // Fetch ticker suggestions while typing
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (query.length > 1 && shouldSearch) {
        setLoadingSearch(true);
        fetch(`http://localhost:8000/search-ticker?q=${encodeURIComponent(query)}`)
          .then((res) => res.json())
          .then((data) => {
            setResults(data.results || []);
            setLoadingSearch(false);
          })
          .catch(() => setLoadingSearch(false));
      } else {
        setResults([]);
      }
    }, 300);

    return () => clearTimeout(timeout);
  }, [query]);

  // Handle Analyze click
  const handleAnalyze = async () => {
    if (!ticker) {
      setError("Please select a ticker.");
      return;
    }
    setError("");
    setLoadingAnalyze(true);
    setAnalysis(null);
    try {
      const res = await fetch("http://localhost:8000/analyze-and-wait", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company_name: query, ticker }),
      });
      if (!res.ok) throw new Error("Failed to analyze");
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoadingAnalyze(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Credit Risk Analysis</h1>

      {/* Search Bar */}
      <div className="relative mb-6">
        <Input
          type="text"
          placeholder="Search for a company or ticker..."
          value={query}
          onChange={(e) => {setShouldSearch(true); setQuery(e.target.value)}}
        />
        {shouldSearch && loadingSearch && <p className="absolute right-3 top-3 text-gray-400 text-sm">Loading...</p>}
        {shouldSearch && results.length > 0 && (
          <ul className="absolute z-10 bg-white border rounded shadow mt-1 w-full max-h-60 overflow-y-auto">
            {results.map((item) => (
              <li
                key={item.ticker}
                className="p-3 hover:bg-gray-100 cursor-pointer"
                onClick={() => {
                  setQuery(item.name);
                  setTicker(item.ticker);
                  setResults([]);
                  setShouldSearch(false);
                }}
              >
                <span className="font-bold">{item.ticker}</span> - {item.name}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Analyze Button */}
      <Button onClick={handleAnalyze} disabled={loadingAnalyze} className="mb-6">
        {loadingAnalyze ? (
          <>
            <Loader2 className="animate-spin h-4 w-4 mr-2" /> Analyzing...
          </>
        ) : (
          "Analyze"
        )}
      </Button>

      {error && <p className="text-red-500 mb-4">{error}</p>}

      {/* Report Section */}
      {analysis && (
        <Card>
          <CardHeader>
            <CardTitle>{analysis.company_info.company}</CardTitle>
            <p className="text-sm text-gray-500">Analysis Date: {analysis.company_info.analysis_date}</p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-lg font-semibold">Final Fused Score:</p>
                <p className="text-2xl">{analysis.company_info.final_fused_score}</p>
                <p className="text-gray-500">Credit Grade: {analysis.company_info.credit_grade}</p>
              </div>
              <div>
                <p className="text-lg font-semibold">Components:</p>
                <p>Structured Score: {analysis.company_info.structured_score}</p>
                <p>Unstructured Score: {analysis.company_info.unstructured_score}</p>
              </div>
            </div>

            {/* Fusion Explanation */}
            <div className="mt-6">
              <h3 className="font-semibold text-lg mb-2">Fusion Analysis</h3>
              <p className="text-gray-600 whitespace-pre-line">{analysis.fusion_explanation.explanation}</p>
            </div>

            {/* Explainability Report */}
            {analysis.explainability_report && (
              <div className="mt-6">
                <h3 className="font-semibold text-lg mb-2">Explainability Report</h3>
                <p className="text-gray-600 whitespace-pre-line">{analysis.explainability_report}</p>
              </div>
            )}

            {/* Historical Scores */}
            <div className="mt-6">
              <h3 className="font-semibold text-lg mb-2">Historical Scores</h3>
              <pre className="bg-gray-100 p-3 rounded text-sm whitespace-pre-line">
                {analysis.historical_scores_summary}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
