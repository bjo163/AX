import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import { AppSidebar } from "./components/app-sidebar"
import { ChartAreaInteractive } from "./components/chart-area-interactive"
import { DataTable } from "./components/data-table"
import { SectionCards } from "./components/section-cards"
import { SiteHeader } from "./components/site-header"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { ThemeProvider } from "@/components/theme-provider"
import LLMPage from "./app/llm/page" // Import halaman LLM
import HomePage from "./app/dashboard/page" // Import halaman Dashboard
import data from "./data.json"
import AgentPage from "./app/agent/page"

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <SidebarProvider>
        <Router>
          <AppSidebar variant="inset" />
          <SidebarInset>
            <SiteHeader />
            <div className="flex flex-1 flex-col">
              <Routes>
                {/* Rute untuk halaman Home */}
                <Route
                  path="/"
                  element={
                    <div className="@container/main flex flex-1 flex-col gap-2">
                      <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
                        <SectionCards />
                        <div className="px-4 lg:px-6">
                          <ChartAreaInteractive />
                        </div>
                        <DataTable data={data} />
                      </div>
                    </div>
                  }
                />
                {/* Rute untuk halaman LLM */}
                <Route path="/llm" element={<LLMPage />} />
                <Route path="/agent" element={<AgentPage />} />
              </Routes>
            </div>
          </SidebarInset>
        </Router>
      </SidebarProvider>
    </ThemeProvider>
  )
}

export default App