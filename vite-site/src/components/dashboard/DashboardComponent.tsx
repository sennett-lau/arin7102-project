const DashboardComponent = ({ isDisplayed }: { isDisplayed: boolean }) => {
  return (
    <div className={`flex-1 items-center justify-center ${isDisplayed ? 'flex' : 'hidden'}`}>
      <div className='text-center'>
        <h2 className='text-2xl font-medium mb-4'>Dashboard View</h2>
        <p className='text-[#8e8ea0]'>Analytics dashboard would be displayed here.</p>
      </div>
    </div>
  );
};

export default DashboardComponent; 
