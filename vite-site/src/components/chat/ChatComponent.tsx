import { KeyboardEvent, useState } from 'react';
import { sendMessage } from '@/utils/chatApi';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

export type MessageType = {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
};

// Type for API message history format
export type ChatHistoryMessage = {
  role: 'user' | 'assistant';
  content: string;
};

const ChatComponent = ({ isDisplayed }: { isDisplayed: boolean }) => {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Convert messages to the format expected by the API
  const getMessageHistory = (): ChatHistoryMessage[] => {
    return messages.map((msg) => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.content,
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

  return (
    <>
      <div className={`flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-[#565869] ${isDisplayed ? 'block' : 'hidden'}`}>
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
          <MessageList messages={messages} isLoading={isLoading} />
        )}
      </div>
      
      <div className={`border-t border-[#565869] p-4 ${isDisplayed ? 'block' : 'hidden'}`}>
        <MessageInput 
          input={input}
          setInput={setInput}
          handleSendMessage={handleSendMessage}
          handleKeyDown={handleKeyDown}
          isLoading={isLoading}
        />
      </div>
    </>
  );
};

export default ChatComponent; 