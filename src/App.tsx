import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { DashboardPage } from './pages/DashboardPage'
import { PendingPage } from './pages/PendingPage'

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route
          path="/transactions"
          element={
            <PendingPage
              title="Live Transactions"
              description="Review incoming transactions and their available risk evidence."
            />
          }
        />
        <Route
          path="/network"
          element={
            <PendingPage
              title="Network Explorer"
              description="Explore account relationships and backend-confirmed suspicious patterns."
            />
          }
        />
        <Route
          path="/cases"
          element={
            <PendingPage
              title="Investigation Cases"
              description="Review cases, evidence, history, and analyst feedback."
            />
          }
        />
        <Route
          path="/analytics"
          element={
            <PendingPage
              title="Analytics"
              description="Review measured system and model performance when data is available."
            />
          }
        />
        <Route path="*" element={<PendingPage title="Page not found" description="This page is not available." />} />
      </Routes>
    </AppShell>
  )
}

export default App
