// Mock responses for pharmacy-related questions
const mockResponses: Record<string, string> = {
  default: "I'm a Pharmacy Product Bot. How can I assist you today?",
  greetings: "Hello! I'm here to help with any questions about pharmacy products.",
  painkillers:
    'We have several over-the-counter pain relief options including acetaminophen (Tylenol), ibuprofen (Advil, Motrin), and aspirin. What type of pain are you experiencing?',
  headache:
    'For headaches, I recommend acetaminophen (Tylenol) or ibuprofen (Advil). Make sure to follow the dosage instructions on the packaging.',
  fever:
    'To reduce fever, both acetaminophen (Tylenol) and ibuprofen (Advil) can be effective. Remember to stay hydrated and rest.',
  allergy:
    'For allergies, antihistamines like loratadine (Claritin), cetirizine (Zyrtec), or diphenhydramine (Benadryl) may help relieve symptoms.',
  cold:
    'For cold symptoms, products like Dayquil/Nyquil, Theraflu, or Mucinex can help. These contain combinations of decongestants, pain relievers, and cough suppressants.',
  vitamins:
    'We offer various vitamins including multivitamins, vitamin C, D, B-complex, and more. What specific health concern are you addressing?',
  prescription:
    "For prescription medications, please consult with your doctor. I can provide general information, but specific medications require a healthcare professional's guidance.",
};

// Function to pick the most relevant response based on keywords
const getRelevantResponse = (message: string): string => {
  const lowerMessage = message.toLowerCase();
  
  if (/hello|hi|hey|greetings/i.test(lowerMessage)) {
    return mockResponses.greetings;
  }
  
  if (/pain|painkiller|hurt/i.test(lowerMessage)) {
    return mockResponses.painkillers;
  }
  
  if (/headache|head pain|migraine/i.test(lowerMessage)) {
    return mockResponses.headache;
  }
  
  if (/fever|temperature|hot/i.test(lowerMessage)) {
    return mockResponses.fever;
  }
  
  if (/allergy|allergic|hay fever|sneez/i.test(lowerMessage)) {
    return mockResponses.allergy;
  }
  
  if (/cold|flu|cough|congestion|runny nose/i.test(lowerMessage)) {
    return mockResponses.cold;
  }
  
  if (/vitamin|supplement|mineral/i.test(lowerMessage)) {
    return mockResponses.vitamins;
  }
  
  if (/prescription|doctor|prescribed/i.test(lowerMessage)) {
    return mockResponses.prescription;
  }
  
  return mockResponses.default;
};

// This would be the actual API call in a real implementation
export const sendMessage = async (message: string): Promise<string> => {
  // Check if mock mode is enabled from environment variables
  const isMock = import.meta.env.VITE_IS_MOCK === 'true';
  
  if (isMock) {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 1000));
    return getRelevantResponse(message);
  } else {
    // This would be the real API call to the backend
    try {
      const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:7101';
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          // This simulates what a backend calling OpenAI might expect
          model: 'gpt-3.5-turbo',
          temperature: 0.7,
          max_tokens: 150,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get response from API');
      }
      
      const data = await response.json();
      return data.reply;
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  }
}; 