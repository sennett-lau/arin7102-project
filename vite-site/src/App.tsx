import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import Layout from '@/components/commons/Layout';
import ChatPage from '@/pages/ChatPage';

import './App.css';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path='/*' element={<ChatPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
