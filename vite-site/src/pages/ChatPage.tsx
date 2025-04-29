import { useState } from 'react';
import ChatComponent from '@/components/chat/ChatComponent';
import DashboardComponent from '@/components/dashboard/DashboardComponent';

const ChatPage = () => {
  const [viewMode, setViewMode] = useState<'chat' | 'dashboard'>('chat');

  const toggleView = () => {
    setViewMode(viewMode === 'chat' ? 'dashboard' : 'chat');
  };

  return (
    <div className='flex flex-col h-screen bg-[#343541] text-white'>
      {/* Header */}
      <div className='flex items-center justify-between p-3 border-b border-[#565869]'>
        <div className='text-sm font-medium text-[#c5c5d2]'>
          ARIN7102 Project | Pharmacy Product Analysis
        </div>
        <button
          onClick={toggleView}
          className='px-3 py-1.5 bg-[#444654] rounded-md text-sm text-white hover:bg-[#565869] transition-colors'
        >
          {viewMode === 'chat' ? 'Dashboard' : 'Chat'}
        </button>
      </div>

      <DashboardComponent isDisplayed={viewMode === 'dashboard'} />
      <ChatComponent isDisplayed={viewMode === 'chat'} />
    </div>
  );
};

export default ChatPage; 
