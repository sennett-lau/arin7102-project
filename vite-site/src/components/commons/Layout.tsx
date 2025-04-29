import { ReactNode } from 'react';

type Props = {
  children: ReactNode;
};

const Layout = (props: Props) => {
  const { children } = props;
  return <div className='min-h-screen bg-[#343541] text-white'>{children}</div>;
};

export default Layout;
