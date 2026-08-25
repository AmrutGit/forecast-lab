import { Header } from "./components/Header";
import { Dashboard } from "./pages/Dashboard";

export default function App() {
  return (
    <div className="app-shell">
      <Header />
      <main className="app-main">
        <Dashboard />
      </main>
    </div>
  );
}
