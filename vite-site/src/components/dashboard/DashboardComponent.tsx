import React, { useState, useRef, useEffect } from 'react';
import MockConversationDashboard from './MockConversationDashboard';

// Define dashboard types
interface Dashboard {
  id: string;
  name: string;
  component: React.ReactNode;
}

// MedicinePriceRange component for displaying the images in a 3x3 grid
const MedicinePriceRange = () => {
  // Array of images from g1 to g9
  const images = Array.from({ length: 9 }, (_, i) => `/assets/price-range/g${i + 1}.png`);

  return (
    <div className='p-4 h-full overflow-y-auto'>
      <h3 className='text-xl font-medium mb-4'>Medicine Price Range</h3>
      <div className='grid grid-cols-3 gap-4'>
        {images.map((image, index) => (
          <div key={index} className='bg-[#444654] rounded-lg overflow-hidden'>
            <img src={image} alt={`Price Range Graph ${index + 1}`} className='w-full h-auto' />
          </div>
        ))}
      </div>
    </div>
  );
};

// SalesTrend component for displaying monthly and weekly sales trends
const SalesTrend = () => {
  const [activeTab, setActiveTab] = useState<'monthly' | 'weekly'>('monthly');
  const overallRef = useRef<HTMLDivElement>(null);
  const perResultRef = useRef<HTMLDivElement>(null);
  
  // Images paths based on selected tab
  const getImages = () => {
    const basePath = `/assets/sales-trend/${activeTab}`;
    return [
      {
        src: `${basePath}/overall.png`,
        alt: `${activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Overall Sales Trend`,
      },
      {
        src: `${basePath}/per-result.png`,
        alt: `${activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Per Result Sales Trend`,
      },
    ];
  };

  // Scroll handling functions
  const scrollToOverall = () => {
    overallRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToPerResult = () => {
    perResultRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className='h-full flex flex-col relative'>
      <div className='p-4 flex-shrink-0'>
        <h3 className='text-xl font-medium mb-4'>Sales Trend</h3>
        
        {/* Tab Navigation */}
        <div className='flex space-x-2 mb-4'>
          <button
            onClick={() => setActiveTab('monthly')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'monthly'
                ? 'bg-[#444654] text-white'
                : 'bg-[#2a2b36] text-[#c5c5d2] hover:bg-[#3a3b47]'
            }`}
          >
            Monthly
          </button>
          <button
            onClick={() => setActiveTab('weekly')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'weekly'
                ? 'bg-[#444654] text-white'
                : 'bg-[#2a2b36] text-[#c5c5d2] hover:bg-[#3a3b47]'
            }`}
          >
            Weekly
          </button>
        </div>
      </div>
      
      {/* Quick Navigation Buttons (Fixed on the right) */}
      <div className='absolute right-4 top-16 transform -translate-y-1/2 flex flex-col space-y-2 z-10'>
        <button
          onClick={scrollToOverall}
          className='px-3 py-2 bg-[#444654] rounded-md text-white shadow-lg hover:bg-[#565869] transition-colors text-sm'
        >
          Overall
        </button>
        <button
          onClick={scrollToPerResult}
          className='px-3 py-2 bg-[#444654] rounded-md text-white shadow-lg hover:bg-[#565869] transition-colors text-sm'
        >
          Per-Result
        </button>
      </div>
      
      {/* Images Container - Scrollable */}
      <div className='flex-1 overflow-y-auto px-4 pb-4'>
        <div className='space-y-6'>
          {getImages().map((image, index) => (
            <div 
              key={index} 
              className='bg-[#444654] rounded-lg overflow-hidden'
              ref={index === 0 ? overallRef : perResultRef}
            >
              <h4 className='p-2 text-sm font-medium text-[#c5c5d2] bg-[#3a3b47]'>
                {index === 0 ? 'Overall Trend' : 'Per Result Trend'}
              </h4>
              <div className='p-4'>
                <img src={image.src} alt={image.alt} className='w-full h-auto' />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// WordCloudDashboard component
const WordCloudDashboard = () => {
  // View types: word clouds or sentiment analysis
  const [viewType, setViewType] = useState<'wordcloud' | 'sentiment'>('wordcloud');
  // Drug selection for filtering the word clouds/sentiment
  const [selectedDrug, setSelectedDrug] = useState<string>('all');
  // Reference to different sections for quick navigation
  const wordCloudRef = useRef<HTMLDivElement>(null);
  const sentimentRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Drugs data
  const drugs = [
    { id: 'all', name: 'All Drugs' },
    { id: 'lexapro', name: 'Lexapro' },
    { id: 'lisinopril', name: 'Lisinopril' },
    { id: 'hydrocodone', name: 'Hydrocodone' },
    { id: 'cymbalta', name: 'Cymbalta' },
  ];
  
  // Gender filters (only for Lexapro)
  const genderFilters = [
    { id: 'all', name: 'All' },
    { id: 'male', name: 'Male' },
    { id: 'female', name: 'Female' },
  ];
  
  const [selectedGender, setSelectedGender] = useState<string>('all');
  
  // Scroll to top when filters change
  useEffect(() => {
    containerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [viewType, selectedDrug, selectedGender]);
  
  // Get image path based on selections
  const getImagePath = () => {
    const basePath = `/assets/word-cloud/${viewType === 'wordcloud' ? 'reviews' : 'sentiment'}`;
    
    if (selectedDrug === 'all' && viewType === 'wordcloud') {
      return '/assets/word-cloud/drugs/all_drugs.png';
    }
    
    // Handle gender filters for Lexapro
    if (selectedDrug === 'lexapro' && selectedGender !== 'all') {
      return `${basePath}/lexapro_${selectedGender}.png`;
    }
    
    return `${basePath}/${selectedDrug}.png`;
  };
  
  // Scroll to section functions
  const scrollToWordCloud = () => {
    wordCloudRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const scrollToSentiment = () => {
    sentimentRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  // Handle filter changes with auto-scroll to top
  const handleViewTypeChange = (type: 'wordcloud' | 'sentiment') => {
    setViewType(type);
  };
  
  const handleDrugChange = (drugId: string) => {
    setSelectedDrug(drugId);
    if (drugId !== 'lexapro') {
      setSelectedGender('all');
    }
  };
  
  const handleGenderChange = (genderId: string) => {
    setSelectedGender(genderId);
  };
  
  return (
    <div className='h-full flex flex-col relative'>
      {/* Header with filters - Fixed */}
      <div className='p-4 flex-shrink-0 bg-[#343541] border-b border-[#565869]'>
        <h3 className='text-xl font-medium mb-3'>Drug Word Cloud Analysis</h3>
        
        <div className='flex flex-wrap gap-4 mb-3'>
          {/* Main Tab Navigation */}
          <div className='flex items-center'>
            <span className='text-sm text-[#c5c5d2] mr-2'>View:</span>
            <div className='flex space-x-1'>
              <button
                onClick={() => handleViewTypeChange('wordcloud')}
                className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
                  viewType === 'wordcloud'
                    ? 'bg-[#444654] text-white'
                    : 'bg-[#2a2b36] text-[#c5c5d2] hover:bg-[#3a3b47]'
                }`}
              >
                Word Clouds
              </button>
              <button
                onClick={() => handleViewTypeChange('sentiment')}
                className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
                  viewType === 'sentiment'
                    ? 'bg-[#444654] text-white'
                    : 'bg-[#2a2b36] text-[#c5c5d2] hover:bg-[#3a3b47]'
                }`}
              >
                Sentiment
              </button>
            </div>
          </div>
          
          {/* Drug Selection Filter */}
          <div className='flex items-center'>
            <span className='text-sm text-[#c5c5d2] mr-2'>Drug:</span>
            <div className='flex flex-wrap gap-1'>
              {drugs.map((drug) => (
                <button
                  key={drug.id}
                  onClick={() => handleDrugChange(drug.id)}
                  className={`px-2 py-1 rounded-md text-xs transition-colors ${
                    selectedDrug === drug.id
                      ? 'bg-[#444654] text-white'
                      : 'bg-[#2a2b36] text-[#c5c5d2] hover:bg-[#3a3b47]'
                  }`}
                >
                  {drug.name}
                </button>
              ))}
            </div>
          </div>
          
          {/* Gender Filter (only for Lexapro) */}
          {selectedDrug === 'lexapro' && (
            <div className='flex items-center'>
              <span className='text-sm text-[#c5c5d2] mr-2'>Gender:</span>
              <div className='flex flex-wrap gap-1'>
                {genderFilters.map((gender) => (
                  <button
                    key={gender.id}
                    onClick={() => handleGenderChange(gender.id)}
                    className={`px-2 py-1 rounded-md text-xs transition-colors ${
                      selectedGender === gender.id
                        ? 'bg-[#444654] text-white'
                        : 'bg-[#2a2b36] text-[#c5c5d2] hover:bg-[#3a3b47]'
                    }`}
                  >
                    {gender.name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Navigation Buttons */}
      <div className='absolute right-4 top-16 transform -translate-y-1/2 flex flex-col space-y-2 z-10'>
        <button
          onClick={scrollToWordCloud}
          className='px-3 py-2 bg-[#444654] rounded-md text-white shadow-lg hover:bg-[#565869] transition-colors text-sm'
        >
          Word Cloud
        </button>
        <button
          onClick={scrollToSentiment}
          className='px-3 py-2 bg-[#444654] rounded-md text-white shadow-lg hover:bg-[#565869] transition-colors text-sm'
        >
          Sentiment
        </button>
      </div>
      
      {/* Content Container - Scrollable */}
      <div ref={containerRef} className='flex-1 overflow-y-auto'>
        <div className='p-4 max-w-4xl mx-auto'>
          {/* Word Cloud Section */}
          <div ref={wordCloudRef} className='bg-[#444654] rounded-lg overflow-hidden mb-6'>
            <h4 className='p-2 text-sm font-medium text-[#c5c5d2] bg-[#3a3b47]'>
              {viewType === 'wordcloud' ? 'Word Cloud Visualization' : 'Sentiment Analysis'}
            </h4>
            <div className='p-4 bg-white'>
              <img src={getImagePath()} alt={`${selectedDrug} ${viewType}`} className='w-full h-auto' />
            </div>
          </div>
          
          {/* Explanation Section */}
          <div ref={sentimentRef} className='bg-[#444654] rounded-lg overflow-hidden'>
            <h4 className='p-2 text-sm font-medium text-[#c5c5d2] bg-[#3a3b47]'>
              Analysis Explanation
            </h4>
            <div className='p-4 text-white'>
              {viewType === 'wordcloud' ? (
                <div>
                  <p className='mb-3'>
                    Word clouds visually represent the frequency of words in drug reviews. 
                    The size of each word indicates how frequently it appears in the reviews.
                  </p>
                  <p className='mb-3'>
                    Common words and stopwords (like &quot;the&quot;, &quot;and&quot;, etc.) are removed to focus on meaningful terms.
                  </p>
                  <p>
                    This visualization helps identify common symptoms, side effects, and patient experiences mentioned in relation to{' '}
                    {selectedDrug === 'all' ? 'all drugs' : selectedDrug}
                    {selectedDrug === 'lexapro' && selectedGender !== 'all' 
                      ? ` for ${selectedGender} patients`
                      : ''}.
                  </p>
                </div>
              ) : (
                <div>
                  <p className='mb-3'>
                    Sentiment analysis evaluates the emotional tone of drug reviews, 
                    categorizing sentiments as positive, negative, or neutral.
                  </p>
                  <p className='mb-3'>
                    The bar chart shows the distribution of sentiment scores for reviews about
                    {selectedDrug === 'all' ? ' all drugs' : ` ${selectedDrug}`}
                    {selectedDrug === 'lexapro' && selectedGender !== 'all' 
                      ? ` from ${selectedGender} patients`
                      : ''}.
                  </p>
                  <p>
                    This analysis helps understand patient satisfaction and experiences with medications.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const DashboardComponent = ({ isDisplayed }: { isDisplayed: boolean }) => {
  // Define available dashboards
  const dashboards: Dashboard[] = [
    {
      id: 'mock-conversations',
      name: 'Mock Conversations',
      component: <MockConversationDashboard />,
    },
    {
      id: 'medicine-price-range',
      name: 'Medicine Price Range',
      component: <MedicinePriceRange />,
    },
    {
      id: 'sales-trend',
      name: 'Sales Trend',
      component: <SalesTrend />,
    },
    {
      id: 'word-cloud',
      name: 'Word Cloud Analysis',
      component: <WordCloudDashboard />,
    },
    // Add more dashboards here in the future
  ];

  // State to track the selected dashboard
  const [selectedDashboard, setSelectedDashboard] = useState<string>(dashboards[0]?.id || '');

  // Find the current dashboard component to display
  const currentDashboard = dashboards.find((d) => d.id === selectedDashboard);

  return (
    <div className={`flex-1 flex overflow-hidden ${isDisplayed ? 'flex' : 'hidden'}`}>
      {/* Left Sidebar with dashboard list - independently scrollable */}
      <div className='w-64 bg-[#2a2b36] border-r border-[#565869] overflow-y-auto h-full'>
        <div className='p-4'>
          <h2 className='text-lg font-medium mb-4'>Dashboards</h2>
          <ul className='space-y-2'>
            {dashboards.map((dashboard) => (
              <li key={dashboard.id}>
                <button
                  onClick={() => setSelectedDashboard(dashboard.id)}
                  className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                    selectedDashboard === dashboard.id
                      ? 'bg-[#444654] text-white'
                      : 'text-[#c5c5d2] hover:bg-[#3a3b47]'
                  }`}
                >
                  {dashboard.name}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Right panel with dashboard content - independently scrollable */}
      <div className='flex-1 bg-[#343541] h-full'>
        {currentDashboard ? (
          currentDashboard.component
        ) : (
          <div className='flex items-center justify-center h-full'>
            <p className='text-[#8e8ea0]'>Select a dashboard to view</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardComponent;
