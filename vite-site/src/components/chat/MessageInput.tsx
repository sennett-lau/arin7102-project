import { KeyboardEvent, Dispatch, SetStateAction } from 'react';

interface MessageInputProps {
  input: string;
  setInput: Dispatch<SetStateAction<string>>;
  handleSendMessage: () => Promise<void>;
  handleKeyDown: (e: KeyboardEvent) => void;
  isLoading: boolean;
}

const MessageInput = ({
  input,
  setInput,
  handleSendMessage,
  handleKeyDown,
  isLoading,
}: MessageInputProps) => {
  return (
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
  );
};

export default MessageInput; 
