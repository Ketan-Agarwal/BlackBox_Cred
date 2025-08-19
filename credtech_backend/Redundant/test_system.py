"""
Test script for CredTech Backend system.
This script runs basic tests to ensure the system is working correctly.
"""
import requests
import json
import time
from datetime import datetime


class CredTechTester:
    """Test class for CredTech Backend API."""
    
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def test_health_check(self):
        """Test the health check endpoint."""
        print("🔍 Testing health check endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_companies_list(self):
        """Test the companies list endpoint."""
        print("🔍 Testing companies list endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/companies")
            if response.status_code == 200:
                companies = response.json()
                print(f"✅ Companies list retrieved: {len(companies)} companies")
                
                if companies:
                    sample_company = companies[0]
                    print(f"   Sample: {sample_company.get('symbol')} - {sample_company.get('name')}")
                    return sample_company.get('id')
                return None
            else:
                print(f"❌ Companies list failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Companies list error: {e}")
            return None
    
    def test_structured_score(self, company_id):
        """Test the structured score endpoint."""
        print(f"🔍 Testing structured score for company {company_id}...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/companies/{company_id}/scores/structured")
            if response.status_code == 200:
                data = response.json()
                score = data.get('structured_score', 0)
                print(f"✅ Structured score retrieved: {score:.2f}")
                print(f"   KMV Distance-to-Default: {data.get('kmv_distance_to_default', 0):.3f}")
                print(f"   Altman Z-Score: {data.get('altman_z_score', 0):.3f}")
                return True
            else:
                print(f"❌ Structured score failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Structured score error: {e}")
            return False
    
    def test_unstructured_score(self, company_id):
        """Test the unstructured score endpoint."""
        print(f"🔍 Testing unstructured score for company {company_id}...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/companies/{company_id}/scores/unstructured")
            if response.status_code == 200:
                data = response.json()
                score = data.get('unstructured_score', 0)
                articles = data.get('articles_analyzed', 0)
                print(f"✅ Unstructured score retrieved: {score:.2f}")
                print(f"   Articles analyzed: {articles}")
                return True
            else:
                print(f"❌ Unstructured score failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Unstructured score error: {e}")
            return False
    
    def test_final_score(self, company_id):
        """Test the final score endpoint."""
        print(f"🔍 Testing final score for company {company_id}...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/companies/{company_id}/scores/final")
            if response.status_code == 200:
                data = response.json()
                final_score = data.get('final_score', 0)
                credit_grade = data.get('credit_grade', 'N/A')
                vix = data.get('market_context', {}).get('current_vix', 0)
                print(f"✅ Final score retrieved: {final_score:.2f} ({credit_grade})")
                print(f"   Current VIX: {vix:.2f}")
                return True
            else:
                print(f"❌ Final score failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Final score error: {e}")
            return False
    
    def test_comprehensive_explanation(self, company_id):
        """Test the comprehensive explanation endpoint."""
        print(f"🔍 Testing comprehensive explanation for company {company_id}...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/companies/{company_id}/explanation")
            if response.status_code == 200:
                data = response.json()
                company_name = data.get('company_name', 'Unknown')
                final_score = data.get('fusion_process', {}).get('final_score', 0)
                credit_grade = data.get('fusion_process', {}).get('credit_grade', 'N/A')
                
                print(f"✅ Comprehensive explanation retrieved for {company_name}")
                print(f"   Final Score: {final_score:.2f} ({credit_grade})")
                
                # Check key components
                if 'plain_language_summary' in data:
                    print("   ✓ Plain language summary included")
                if 'structured_model' in data:
                    print("   ✓ Structured model explanation included")
                if 'unstructured_model' in data:
                    print("   ✓ Unstructured model explanation included")
                if 'fusion_process' in data:
                    print("   ✓ Fusion process explanation included")
                if 'trend_analysis' in data:
                    print("   ✓ Trend analysis included")
                
                return True
            else:
                print(f"❌ Comprehensive explanation failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Comprehensive explanation error: {e}")
            return False
    
    def test_sentiment_analysis(self):
        """Test the sentiment analysis endpoint."""
        print("🔍 Testing sentiment analysis endpoint...")
        
        test_texts = [
            "Company reports record profits and strong growth outlook",
            "Stock price falls amid concerns about future performance",
            "Company maintains steady revenue with no major changes"
        ]
        
        try:
            for text in test_texts:
                response = self.session.post(
                    f"{self.base_url}/api/sentiment/analyze",
                    params={"text": text}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data.get('sentiment_analysis', {})
                    label = sentiment.get('predicted_class', 'unknown')
                    score = sentiment.get('processed_score', 0)
                    print(f"   Text: '{text[:50]}...'")
                    print(f"   Sentiment: {label} (score: {score:.1f})")
                else:
                    print(f"❌ Sentiment analysis failed for text: {response.status_code}")
                    return False
            
            print("✅ Sentiment analysis tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Sentiment analysis error: {e}")
            return False
    
    def test_add_company(self):
        """Test adding a new company."""
        print("🔍 Testing add company endpoint...")
        
        test_company = {
            "symbol": "TEST",
            "name": "Test Company Inc.",
            "sector": "Technology",
            "industry": "Software Testing"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/companies",
                params=test_company
            )
            
            if response.status_code == 200:
                data = response.json()
                company_id = data.get('company_id')
                print(f"✅ Company added successfully: {test_company['symbol']} (ID: {company_id})")
                return company_id
            else:
                print(f"❌ Add company failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Add company error: {e}")
            return None
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("=" * 60)
        print("  CredTech Backend - System Tests")
        print("=" * 60)
        print()
        
        # Test 1: Health Check
        health_ok = self.test_health_check()
        print()
        
        if not health_ok:
            print("❌ Health check failed. Cannot proceed with other tests.")
            return False
        
        # Test 2: Companies List
        company_id = self.test_companies_list()
        print()
        
        if not company_id:
            print("❌ No companies found. Cannot proceed with score tests.")
            return False
        
        # Test 3: Structured Score
        structured_ok = self.test_structured_score(company_id)
        print()
        
        # Test 4: Unstructured Score
        unstructured_ok = self.test_unstructured_score(company_id)
        print()
        
        # Test 5: Final Score
        final_ok = self.test_final_score(company_id)
        print()
        
        # Test 6: Comprehensive Explanation
        explanation_ok = self.test_comprehensive_explanation(company_id)
        print()
        
        # Test 7: Sentiment Analysis
        sentiment_ok = self.test_sentiment_analysis()
        print()
        
        # Test 8: Add Company
        new_company_id = self.test_add_company()
        print()
        
        # Results Summary
        print("=" * 60)
        print("  Test Results Summary")
        print("=" * 60)
        
        tests = [
            ("Health Check", health_ok),
            ("Companies List", company_id is not None),
            ("Structured Score", structured_ok),
            ("Unstructured Score", unstructured_ok),
            ("Final Score", final_ok),
            ("Comprehensive Explanation", explanation_ok),
            ("Sentiment Analysis", sentiment_ok),
            ("Add Company", new_company_id is not None)
        ]
        
        passed = sum(1 for _, result in tests if result)
        total = len(tests)
        
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print()
        print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! System is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the logs for details.")
            return False


def main():
    """Main test function."""
    tester = CredTechTester()
    
    print("Starting CredTech Backend system tests...")
    print("Make sure the server is running at http://127.0.0.1:8000")
    print()
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("The CredTech Backend system is ready for the hackathon!")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and try again.")
    
    return success


if __name__ == "__main__":
    main()
