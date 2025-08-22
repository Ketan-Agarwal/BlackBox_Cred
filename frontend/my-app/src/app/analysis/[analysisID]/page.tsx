"use client";

import { useEffect, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface Report {
  company_info: {
    company: string;
    credit_grade: string;
    analysis_date: string;
    structured_score: string;
    final_fused_score: string;
    unstructured_score: string;
  };
  fusion_explanation: {
    market_regime: string;
    expert_agreement: number;
    dynamic_weights: { structured_expert: number; news_sentiment_expert: number };
    expert_contributions: Record<string, { weight: number; risk_score: number; contribution: number }>;
  };
  explainability_report: string;
  historical_scores_detailed: Record<string, { date: string; final_score: number; credit_grade: string }>;
}

export default function AnalysisDashboard() {
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/mock-report.json") // Replace with real API
      .then((res) => res.json())
      .then((data) => {
        setReport(data);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <Loader2 className="animate-spin h-8 w-8" />
      </div>
    );
  }

  if (!report) return <p>No data available.</p>;

  const { company_info, fusion_explanation, explainability_report, historical_scores_detailed } = report;

  const chartData = {
    series: [{ name: "Fused Score", data: Object.values(historical_scores_detailed).map((d) => d.final_score) }],
    options: { chart: { type: "line" }, xaxis: { categories: Object.values(historical_scores_detailed).map((d) => d.date) } },
  };

  // Extract data for drivers and news
  const drivers = explainability_report
    .split("\n")
    .filter((line) => line.match(/^\s*\d+\./))
    .map((line) => line.trim());

  const news = explainability_report
    .split("\n")
    .filter((line) => line.includes("Sentiment:"))
    .map((line) => {
      const sentiment = line.includes("Positive") ? "positive" : line.includes("Negative") ? "negative" : "neutral";
      return { text: line, sentiment };
    });

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold">{company_info.company}</h1>
        <p className="text-gray-500 text-lg">Credit Grade: {company_info.credit_grade}</p>
        <p className="text-sm text-gray-400">Analysis Date: {company_info.analysis_date}</p>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview">
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="drivers">Explainability</TabsTrigger>
          <TabsTrigger value="news">News & Sentiment</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card><CardHeader><CardTitle>Final Fused Score</CardTitle></CardHeader><CardContent className="text-2xl font-bold">{company_info.final_fused_score}</CardContent></Card>
            <Card><CardHeader><CardTitle>Structured Score</CardTitle></CardHeader><CardContent>{company_info.structured_score}</CardContent></Card>
            <Card><CardHeader><CardTitle>Unstructured Score</CardTitle></CardHeader><CardContent>{company_info.unstructured_score}</CardContent></Card>
          </div>

          <Card>
            <CardHeader><CardTitle>Fusion Summary</CardTitle></CardHeader>
            <CardContent>
              <p>Market Regime: {fusion_explanation.market_regime}</p>
              <p>Expert Agreement: {(fusion_explanation.expert_agreement * 100).toFixed(1)}%</p>
              <div className="flex gap-4 mt-4">
                <div className="flex-1">
                  <p className="text-sm">Structured Expert</p>
                  <Progress value={fusion_explanation.dynamic_weights.structured_expert * 100} />
                </div>
                <div className="flex-1">
                  <p className="text-sm">News Expert</p>
                  <Progress value={fusion_explanation.dynamic_weights.news_sentiment_expert * 100} />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Expert Contributions</CardTitle></CardHeader>
            <CardContent>
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border px-4 py-2">Expert</th>
                    <th className="border px-4 py-2">Score</th>
                    <th className="border px-4 py-2">Weight</th>
                    <th className="border px-4 py-2">Contribution</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(fusion_explanation.expert_contributions).map(([key, val]) => (
                    <tr key={key}>
                      <td className="border px-4 py-2">{key}</td>
                      <td className="border px-4 py-2">{val.risk_score.toFixed(1)}</td>
                      <td className="border px-4 py-2">{(val.weight * 100).toFixed(1)}%</td>
                      <td className="border px-4 py-2">{val.contribution.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Drivers Tab */}
        <TabsContent value="drivers">
          <Card>
            <CardHeader><CardTitle>Key Financial Drivers</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {drivers.map((d, i) => (
                  <li key={i} className="p-2 border rounded">{d}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        {/* News Tab */}
        <TabsContent value="news">
          <Card>
            <CardHeader><CardTitle>News Impact</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {news.map((n, i) => (
                  <li key={i} className="flex items-center gap-2 p-2 border rounded">
                    <Badge variant={
                      n.sentiment === "positive" ? "success" :
                      n.sentiment === "negative" ? "destructive" : "secondary"
                    }>
                      {n.sentiment}
                    </Badge>
                    {n.text}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history">
          <Card>
            <CardHeader><CardTitle>Historical Scores</CardTitle></CardHeader>
            <CardContent>
              <Chart options={chartData.options} series={chartData.series} type="line" height={300} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
