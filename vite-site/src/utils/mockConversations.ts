// Mock conversation data for demonstration purposes
import { ChatHistoryMessage } from '@/components/chat/ChatComponent';

type MockConversation = {
  messages: ChatHistoryMessage[];
};

// MOCK_CONVERSATION_INDEX = 1: Drug recommendation for fever
const conversation1: MockConversation = {
  messages: [
    {
      role: 'user',
      content: 'Please recommend a drug to sell for a male customer with age of 35 and having fever',
    },
    {
      role: 'assistant',
      content: 'For a 35-year-old male with fever, I would recommend acetaminophen (Tylenol) or ibuprofen (Advil). Both are effective in reducing fever and relieving associated discomfort. Make sure to advise the customer to follow the recommended dosage on the packaging and to stay hydrated.',
    },
  ],
};

// MOCK_CONVERSATION_INDEX = 2: Information about Cymbalta
const conversation2: MockConversation = {
  messages: [
    {
      role: 'user',
      content: 'Please let me know more about cymbalta',
    },
    {
      role: 'assistant',
      content: `Summary of reviews for cymbalta:

**Effectiveness**
Cymbalta is a good drug for depression and anxiety and it is effective in reducing the pain in their legs and feet and improving their mood

**Side Effects**
Users have been on Cymbalta for 6 months and Users have had a lot of side effects that are very serious and cause Users to be very tired and dizzy

**Overall Experience**
Users have been taking this drug for 4 months and it has been working well for Users with no side effects at all. Users am very depressed and can not sleep.`,
    },
  ],
};

// MOCK_CONVERSATION_INDEX = 3: Unknown drug
const conversation3: MockConversation = {
  messages: [
    {
      role: 'user',
      content: 'Please let me know more about xeonoticlipsis',
    },
    {
      role: 'assistant',
      content: `I apologize, but there is no information about "xeonoticlipsis" in our knowledge base. This appears to be an unknown or non-existent medication. If you're looking for information about a specific medication, please double-check the spelling or provide more details.`,
    },
  ],
};

// Export all conversations indexed
export const mockConversations: Record<number, MockConversation> = {
  1: conversation1,
  2: conversation2,
  3: conversation3,
};

// Function to get a conversation by index
export const getMockConversation = (index: number): MockConversation | undefined => {
  return mockConversations[index];
}; 