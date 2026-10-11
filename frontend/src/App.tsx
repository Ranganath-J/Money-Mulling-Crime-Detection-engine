import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { DashboardPage } from './pages/DashboardPage'
import { NetworkExplorerPage } from './pages/NetworkExplorerPage'
import { PendingPage } from './pages/PendingPage'
import { TransactionsPage } from './pages/TransactionsPage'

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route
          path="/transactions"
          element={<TransactionsPage />}
        />
        <Route
          path="/network"
          element={<NetworkExplorerPage />}
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
