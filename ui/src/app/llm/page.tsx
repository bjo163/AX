'use client'

import { useState, useEffect } from "react"
import axios from "axios"
import { AppSidebar } from "../../components/app-sidebar"
import { SiteHeader } from "../../components/site-header"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Bot, Send, Settings, Sparkles, User, Loader2 } from 'lucide-react'

// Define the model interface based on the API response
interface LLMModel {
  name: string
  description: string
  version: string
  author: string
}

export default function LLMPage() {
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([
    { role: "assistant", content: "Hello! How can I help you today?" },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [models, setModels] = useState<LLMModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [isLoadingModels, setIsLoadingModels] = useState(true)
  const [modelError, setModelError] = useState<string | null>(null)

  // Fetch models from API when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setIsLoadingModels(true)
        setModelError(null)
        
        const response = await axios.get('http://localhost:8000/api/v1/llms')
        
        // Axios automatically throws for non-2xx responses, so we don't need to check response.ok
        const data = response.data
        setModels(data.data)
        
        // Set the first model as selected by default if available
        if (data.data.length > 0) {
          setSelectedModel(data.data[0].name)
        }
      } catch (error) {
        console.error('Error fetching models:', error)
        
        // Axios specific error handling
        if (axios.isAxiosError(error)) {
          setModelError(
            error.response 
              ? `Error: ${error.response.status} - ${error.response.statusText}` 
              : error.message
          )
        } else {
          setModelError('Failed to fetch models')
        }
      } finally {
        setIsLoadingModels(false)
      }
    }

    fetchModels()
  }, [])

  const handleSend = () => {
    if (!input.trim()) return

    // Add user message
    const newMessages = [...messages, { role: "user" as const, content: input }]
    setMessages(newMessages)
    setInput("")

    // Simulate AI response
    setIsLoading(true)
    setTimeout(() => {
      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content:
            `This is a simulated response using the ${selectedModel} model. In a real application, this would be connected to the actual API.`,
        },
      ])
      setIsLoading(false)
    }, 1000)
  }

  return (
    <div className="flex flex-1 flex-col">
      <div className="@container/main flex flex-1 flex-col gap-2">
        <div className="flex flex-1 flex-col gap-4 p-4 md:gap-6 md:p-6">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="col-span-2">
              <Card className="flex h-[calc(100vh-12rem)] flex-col">
                <CardHeader className="px-4 py-3 sm:px-6">
                  <CardTitle>AI Chat</CardTitle>
                  <CardDescription>Interact with large language models</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 overflow-auto p-4 sm:p-6">
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`flex max-w-[80%] items-start gap-3 rounded-lg px-4 py-2 ${
                            message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                          }`}
                        >
                          <div className="mt-0.5">
                            {message.role === "user" ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
                          </div>
                          <div className="break-words">{message.content}</div>
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="flex max-w-[80%] items-start gap-3 rounded-lg bg-muted px-4 py-2">
                          <div className="mt-0.5">
                            <Bot className="h-5 w-5" />
                          </div>
                          <div className="animate-pulse">Thinking...</div>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
                <CardFooter className="border-t p-4 sm:p-6">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault()
                      handleSend()
                    }}
                    className="flex w-full items-center gap-2"
                  >
                    <Input
                      placeholder="Type your message..."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      className="flex-1"
                    />
                    <Button type="submit" size="icon" disabled={isLoading}>
                      <Send className="h-4 w-4" />
                      <span className="sr-only">Send</span>
                    </Button>
                  </form>
                </CardFooter>
              </Card>
            </div>
            <div className="col-span-1">
              <Tabs defaultValue="models">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="models">Models</TabsTrigger>
                  <TabsTrigger value="settings">Settings</TabsTrigger>
                </TabsList>
                <TabsContent value="models" className="mt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Select Model</CardTitle>
                      <CardDescription>Choose the AI model to use for your conversation</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        {isLoadingModels ? (
                          <div className="flex items-center justify-center py-4">
                            <Loader2 className="h-6 w-6 animate-spin text-primary" />
                            <span className="ml-2">Loading models...</span>
                          </div>
                        ) : modelError ? (
                          <div className="rounded-md bg-destructive/15 p-3 text-sm text-destructive">
                            {modelError}
                          </div>
                        ) : (
                          <Select 
                            value={selectedModel} 
                            onValueChange={setSelectedModel}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select a model" />
                            </SelectTrigger>
                            <SelectContent>
                              {models.map((model) => (
                                <SelectItem key={model.name} value={model.name}>
                                  <div className="flex flex-col">
                                    <div className="flex items-center gap-2">
                                      <Sparkles className="h-4 w-4" />
                                      <span>{model.name}</span>
                                    </div>
                                    <span className="text-xs text-muted-foreground">
                                      {model.description} (v{model.version}) by {model.author}
                                    </span>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        )}
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Temperature</label>
                        <div className="flex items-center gap-2">
                          <span className="text-xs">Precise</span>
                          <Input type="range" min="0" max="1" step="0.1" defaultValue="0.7" className="flex-1" />
                          <span className="text-xs">Creative</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">Max Tokens</label>
                        <Input type="number" defaultValue="4096" />
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="settings" className="mt-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Chat Settings</CardTitle>
                      <CardDescription>Configure your chat experience</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <label className="text-sm font-medium">System Prompt</label>
                        <Textarea
                          placeholder="You are a helpful AI assistant..."
                          className="min-h-[100px]"
                          defaultValue="You are a helpful AI assistant. You provide clear, concise, and accurate information to the user's questions."
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-medium">API Key</label>
                        <Input type="password" placeholder="sk-..." />
                        <p className="text-xs text-muted-foreground">
                          Your API key is stored locally and never sent to our servers
                        </p>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button className="w-full">
                        <Settings className="mr-2 h-4 w-4" />
                        Save Settings
                      </Button>
                    </CardFooter>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}