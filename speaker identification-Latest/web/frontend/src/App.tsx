import { useEffect, useState } from "react";
import BottomBar from "./components/BottomBar";
import HowItWorks from "./components/HowItWorks";
import Navbar from "./components/Navbar";
import RecordingList from "./components/RecordingList";
import RecordingView from "./components/RecordingView";
import UploadCard from "./components/UploadCard";

/** Pages: #/ (home), #/recordings, #/how-it-works, #/recording/<id> and #/recording/<id>/<stage>. */
function readAddress() {
  const hash = window.location.hash;
  const match = hash.match(/^#\/recording\/([a-z0-9_]+)(?:\/([a-z]+))?$/);
  if (match) return { page: "recording", id: match[1], stage: match[2] ?? null };
  if (hash === "#/recordings") return { page: "recordings", id: null, stage: null };
  if (hash === "#/how-it-works") return { page: "how", id: null, stage: null };
  return { page: "home", id: null, stage: null };
}

export default function App() {
  const [address, setAddress] = useState(readAddress());

  useEffect(() => {
    const onChange = () => {
      const next = readAddress();
      if (next.page !== "recording") window.scrollTo(0, 0);
      setAddress(next);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const open = (id: string) => (window.location.hash = `#/recording/${id}`);
  return (
    <div className="app">
      <Navbar page={address.page} />
      {address.page === "recording" && address.id ? (
        <main className="workspace-main">
          <RecordingView key={address.id} id={address.id} chosen={address.stage} />
        </main>
      ) : address.page === "recordings" ? (
        <main className="page">
          <div className="page-head">
            <h1>Recordings</h1>
            <p className="muted">Every consultation analysed on this computer.</p>
          </div>
          <RecordingList />
        </main>
      ) : address.page === "how" ? (
        <main className="page">
          <HowItWorks />
        </main>
      ) : (
        <main className="page home">
          <UploadCard onStarted={open} />
          <section className="recent">
            <div className="section-head">
              <h2>Recent analyses</h2>
              <a className="more-link" href="#/recordings">
                All recordings →
              </a>
            </div>
            <RecordingList limit={5} />
          </section>
        </main>
      )}
      <BottomBar currentId={address.id} />
    </div>
  );
}
