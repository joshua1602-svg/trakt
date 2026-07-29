import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import clsx from "clsx";

interface ToastMessage {
  id: number;
  text: string;
  tone: "info" | "error" | "success";
}

const ToastContext = createContext<{
  show: (text: string, tone?: ToastMessage["tone"]) => void;
} | null>(null);

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("ToastProvider is missing above this component");
  return ctx;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const counter = useRef(0);

  const show = useCallback((text: string, tone: ToastMessage["tone"] = "info") => {
    const id = ++counter.current;
    setToasts((prev) => [...prev, { id, text, tone }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  }, []);

  return (
    <ToastContext.Provider value={{ show }}>
      {children}
      <div className="pointer-events-none fixed bottom-6 right-6 z-50 flex w-80 flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role="status"
            className={clsx(
              "pointer-events-auto rounded-xl border px-4 py-3 text-sm shadow-lg",
              toast.tone === "error" && "border-rose-200 bg-rose-50 text-rose-800",
              toast.tone === "success" && "border-emerald-200 bg-emerald-50 text-emerald-800",
              toast.tone === "info" && "border-stone-200 bg-white text-stone-700",
            )}
          >
            {toast.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
