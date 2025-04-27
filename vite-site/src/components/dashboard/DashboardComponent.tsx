import React, { useState, useRef } from 'react';

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

const DashboardComponent = ({ isDisplayed }: { isDisplayed: boolean }) => {
  // Define available dashboards
  const dashboards: Dashboard[] = [
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
