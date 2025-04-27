import { MessageType } from './ChatComponent';

interface MessageListProps {
  messages: MessageType[];
  isLoading: boolean;
}

const MessageList = ({ messages, isLoading }: MessageListProps) => {
  return (
    <>
      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-[70%] rounded-lg p-3 ${
              message.sender === 'user' ? 'bg-[#343541] border border-[#565869]' : 'bg-[#444654]'
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
      ))}
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
    </>
  );
};

export default MessageList; 
