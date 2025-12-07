// Frontend configuration for the Physical AI Textbook RAG Chatbot

// Base API URL configuration
const config = {
  // Production URL - Update this to your actual deployed backend URL
  production: 'https://your-actual-backend-url.onrender.com',
  
  // Development URL
  development: 'http://localhost:8000',
  
  // Get the appropriate API base URL based on environment
  getApiBaseUrl() {
    // For static sites deployed to GitHub Pages, we check for the domain
    // In a real deployment scenario, you would set the proper URL based on your environment
    if (typeof window !== 'undefined') {
      // Client-side code
      const isProduction = process.env.NODE_ENV === 'production' || 
                          window.location.hostname !== 'localhost';
      
      // For GitHub Pages deployment, you would replace this with your actual backend URL
      return isProduction 
        ? this.production  // Replace with your actual deployed backend URL
        : this.development;
    }
    
    // Server-side rendering fallback
    return process.env.NODE_ENV === 'production' 
      ? this.production 
      : this.development;
  },
  
  // Function to get the full API endpoint
  getApiEndpoint(endpoint) {
    return `${this.getApiBaseUrl()}${endpoint}`;
  }
};

export default config;