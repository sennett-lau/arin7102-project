import React, { useState, useEffect } from 'react';
import { mockConversations } from '@/utils/mockConversations';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
}

const Message: React.FC<MessageProps> = ({ role, content }) => {
  return (
    <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[70%] rounded-lg p-3 ${
          role === 'user' ? 'bg-[#343541] border border-[#565869]' : 'bg-[#444654]'
        }`}
      >
        <div className="text-xs text-[#8e8ea0] mb-1">{role === 'user' ? 'User' : 'Assistant'}</div>
        <p className="whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  );
};

const MockConversationDashboard: React.FC = () => {
  const [selectedIndex, setSelectedIndex] = useState<number>(1);
  const [currentMockIndex, setCurrentMockIndex] = useState<number | null>(null);

  useEffect(() => {
    // Get the mock conversation index from .env
    const mockIndex = import.meta.env.VITE_MOCK_CONVERSATION_INDEX;
    if (mockIndex) {
      setCurrentMockIndex(parseInt(mockIndex));
      setSelectedIndex(parseInt(mockIndex));
    }
  }, []);

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedIndex(parseInt(e.target.value));
  };

  const conversationOptions = Object.keys(mockConversations).map((key) => ({
    value: key,
    label: `Mock Conversation ${key}`,
  }));

  const selectedConversation = mockConversations[selectedIndex];

  return (
    <div className="p-6 h-full overflow-y-auto">
      <div className="mb-6">
        <h2 className="text-xl font-medium mb-4">Conversation Preview</h2>
        
        <div className="mb-4 flex items-center">
          <label htmlFor="conversation-select" className="mr-3">
            Select Conversation:
          </label>
          <select
            id="conversation-select"
            value={selectedIndex}
            onChange={handleSelectChange}
            className="bg-[#444654] border border-[#565869] rounded-md p-2"
          >
            {conversationOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {currentMockIndex && (
          <div className="mb-4 p-3 bg-[#444654] rounded-lg">
            <p>
              Current active mock conversation index: <strong>{currentMockIndex}</strong>
            </p>
            <p className="text-sm text-[#8e8ea0] mt-1">
              (Set in .env as VITE_MOCK_CONVERSATION_INDEX)
            </p>
          </div>
        )}
      </div>

      <div className="bg-[#2a2b36] rounded-lg p-4">
        <h3 className="text-lg font-medium mb-4 border-b border-[#565869] pb-2">
          Conversation Preview
        </h3>
        
        {selectedConversation ? (
          <div className="space-y-4">
            {selectedConversation.messages.map((message, index) => (
              <Message key={index} role={message.role} content={message.content} />
            ))}
          </div>
        ) : (
          <p className="text-[#8e8ea0]">No conversation selected</p>
        )}
      </div>
    </div>
  );
};

export default MockConversationDashboard; 