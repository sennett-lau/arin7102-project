import { KeyboardEvent, useState } from 'react';

import { sendMessage } from '@/utils/chatApi';

type MessageType = {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
};

// Type for API message history format
type ChatHistoryMessage = {
  role: 'user' | 'assistant';
  content: string;
};

const ChatPage = () => {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState<'chat' | 'dashboard'>('chat');

  // Convert messages to the format expected by the API
  const getMessageHistory = (): ChatHistoryMessage[] => {
    return messages.map(msg => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.content
    }));
  };

  const handleSendMessage = async () => {
    if (input.trim() === '') return;
    
    const userMessage: MessageType = {
      id: Date.now().toString(),
      content: input,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      // Get conversation history for context
      const history = getMessageHistory();
      
      // Call API with current message and history
      const response = await sendMessage(input, history);
      
      const botMessage: MessageType = {
        id: (Date.now() + 1).toString(),
        content: response,
        sender: 'bot',
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: MessageType = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

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

      {viewMode === 'dashboard' ? (
        <div className='flex-1 flex items-center justify-center'>
          <div className='text-center'>
            <h2 className='text-2xl font-medium mb-4'>Dashboard View</h2>
            <p className='text-[#8e8ea0]'>Analytics dashboard would be displayed here.</p>
          </div>
        </div>
      ) : (
        <>
          <div className='flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-[#565869]'>
            {messages.length === 0 ? (
              <div className='flex flex-col items-center justify-center h-full'>
                <h1 className='text-3xl font-medium text-white mb-6'>
                  Pharmacy Product Analytics Bot
                </h1>
                <p className='text-[#8e8ea0] text-lg mb-8 max-w-md text-center'>
                  Ask questions about pharmacy products and get instant answers.
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] rounded-lg p-3 ${
                      message.sender === 'user'
                        ? 'bg-[#343541] border border-[#565869]'
                        : 'bg-[#444654]'
                    }`}
                  >
                    <p className='whitespace-pre-wrap'>{message.content}</p>
                    <p className='text-xs text-[#8e8ea0] mt-1'>
                      {message.timestamp.toLocaleTimeString([], { 
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className='flex justify-start'>
                <div className='bg-[#444654] rounded-lg p-3'>
                  <div className='flex space-x-2'>
                    <div className='w-2 h-2 rounded-full bg-[#8e8ea0] animate-bounce'></div>
                    <div className='w-2 h-2 rounded-full bg-[#8e8ea0] animate-bounce delay-100'></div>
                    <div className='w-2 h-2 rounded-full bg-[#8e8ea0] animate-bounce delay-200'></div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <div className='border-t border-[#565869] p-4'>
            <div className='max-w-3xl mx-auto relative'>
              <textarea
                className='w-full p-3 pr-12 bg-[#40414f] border border-[#565869] rounded-lg resize-none focus:outline-none focus:ring-1 focus:ring-[#8e8ea0] text-white placeholder:text-[#8e8ea0]'
                placeholder='Type your message here...'
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button
                className='absolute right-3 bottom-3 p-1 text-[#8e8ea0] hover:text-white disabled:text-[#565869] transition-colors'
                onClick={handleSendMessage}
                disabled={isLoading || input.trim() === ''}
              >
                <svg 
                  xmlns='http://www.w3.org/2000/svg' 
                  fill='none' 
                  viewBox='0 0 24 24' 
                  strokeWidth={1.5} 
                  stroke='currentColor' 
                  className='w-5 h-5'
                >
                  <path 
                    strokeLinecap='round' 
                    strokeLinejoin='round' 
                    d='M6 12 3.269 3.126A59.768 59.768 0 0 1 21.485 12 59.77 59.77 0 0 1 3.27 20.876L5.999 12Zm0 0h7.5'
                  />
                </svg>
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ChatPage; 