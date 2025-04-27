import React, { useState } from 'react';

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
    <div className='p-4'>
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

const DashboardComponent = ({ isDisplayed }: { isDisplayed: boolean }) => {
  // Define available dashboards
  const dashboards: Dashboard[] = [
    {
      id: 'medicine-price-range',
      name: 'Medicine Price Range',
      component: <MedicinePriceRange />,
    },
    // Add more dashboards here in the future
  ];

  // State to track the selected dashboard
  const [selectedDashboard, setSelectedDashboard] = useState<string>(dashboards[0]?.id || '');

  // Find the current dashboard component to display
  const currentDashboard = dashboards.find((d) => d.id === selectedDashboard);

  return (
    <div className={`flex-1 flex ${isDisplayed ? 'flex' : 'hidden'}`}>
      {/* Left Sidebar with dashboard list */}
      <div className='w-64 bg-[#2a2b36] border-r border-[#565869] overflow-y-auto'>
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

      {/* Right panel with dashboard content */}
      <div className='flex-1 overflow-auto bg-[#343541]'>
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
