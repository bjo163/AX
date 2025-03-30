'use client'

import { useState, useEffect } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Loader2, Save, Trash, Edit } from "lucide-react"

// Define the agent and LLM interfaces
interface Agent {
  id: number
  name: string
  description: string
  goal: string
  llm_id: number | null // Allow null values
  llm_name: string
  configuration: Record<string, any>
  create_date: string
  write_date: string
}

interface LLM {
  id: number
  name: string
}

export default function AgentPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [llms, setLLMs] = useState<LLM[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [form, setForm] = useState({
    id: null,
    name: "",
    description: "",
    goal: "",
    llm_id: "", // Default to an empty string
    configuration: "{}",
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch agents and LLMs when component mounts
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true)
        const [agentsResponse, llmsResponse] = await Promise.all([
          axios.get("http://localhost:8000/api/v1/agents"),
          axios.get("http://localhost:8000/api/v1/llms"),
        ])
        setAgents(agentsResponse.data)
        setLLMs(llmsResponse.data.data)
      } catch (error) {
        console.error("Error fetching data:", error)
        setError("Failed to fetch data")
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSelectChange = (value: string) => {
    setForm((prev) => ({ ...prev, llm_id: value }))
  }

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true)
      setError(null)

      const payload = {
        ...form,
        llm_id: form.llm_id ? parseInt(form.llm_id, 10) : null, // Handle null values
        configuration: JSON.parse(form.configuration),
      }

      if (form.id) {
        // Update agent
        await axios.put(`http://localhost:8000/api/v1/agents/${form.id}`, payload)
      } else {
        // Create agent
        await axios.post("http://localhost:8000/api/v1/agents", payload)
      }

      // Refresh agents list
      const agentsResponse = await axios.get("http://localhost:8000/api/v1/agents")
      setAgents(agentsResponse.data)

      // Reset form
      setForm({
        id: null,
        name: "",
        description: "",
        goal: "",
        llm_id: "",
        configuration: "{}",
      })
    } catch (error) {
      console.error("Error saving agent:", error)
      setError("Failed to save agent")
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleEdit = (agent: Agent) => {
    setForm({
      id: agent.id,
      name: agent.name,
      description: agent.description,
      goal: agent.goal,
      llm_id: agent.llm_id ? agent.llm_id.toString() : "", // Handle null values
      configuration: JSON.stringify(agent.configuration, null, 2),
    })
  }

  const handleDelete = async (id: number) => {
    try {
      await axios.delete(`http://localhost:8000/api/v1/agents/${id}`)
      setAgents((prev) => prev.filter((agent) => agent.id !== id))
    } catch (error) {
      console.error("Error deleting agent:", error)
      setError("Failed to delete agent")
    }
  }

  return (
    <div className="flex flex-1 flex-col">
      <div className="@container/main flex flex-1 flex-col gap-2">
        <div className="flex flex-1 flex-col gap-4 p-4 md:gap-6 md:p-6">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="col-span-2">
              <Card className="flex h-[calc(100vh-12rem)] flex-col">
                <CardHeader className="px-4 py-3 sm:px-6">
                  <CardTitle>Agents</CardTitle>
                  <CardDescription>Manage your agents</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 overflow-auto p-4 sm:p-6">
                  {isLoading ? (
                    <div className="flex items-center justify-center py-4">
                      <Loader2 className="h-6 w-6 animate-spin text-primary" />
                      <span className="ml-2">Loading agents...</span>
                    </div>
                  ) : error ? (
                    <div className="rounded-md bg-destructive/15 p-3 text-sm text-destructive">
                      {error}
                    </div>
                  ) : (
                    <ul className="space-y-4">
                      {agents.map((agent) => (
                        <li key={agent.id} className="flex items-center justify-between">
                          <div>
                            <h4 className="font-medium">{agent.name}</h4>
                            <p className="text-sm text-muted-foreground">{agent.description}</p>
                            <p className="text-xs text-muted-foreground">LLM: {agent.llm_name}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button variant="outline" size="icon" onClick={() => handleEdit(agent)}>
                              <Edit className="h-4 w-4" />
                            </Button>
                            <Button variant="destructive" size="icon" onClick={() => handleDelete(agent.id)}>
                              <Trash className="h-4 w-4" />
                            </Button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            </div>
            <div className="col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle>{form.id ? "Edit Agent" : "Create Agent"}</CardTitle>
                  <CardDescription>Fill in the details to {form.id ? "edit" : "create"} an agent</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    name="name"
                    placeholder="Agent Name"
                    value={form.name}
                    onChange={handleInputChange}
                  />
                  <Textarea
                    name="description"
                    placeholder="Agent Description"
                    value={form.description}
                    onChange={handleInputChange}
                  />
                  <Textarea
                    name="goal"
                    placeholder="Agent Goal"
                    value={form.goal}
                    onChange={handleInputChange}
                  />
                 <Select value={form.llm_id || ""} onValueChange={handleSelectChange}>
  <SelectTrigger>
    <SelectValue placeholder="Select LLM" />
  </SelectTrigger>
  <SelectContent>
    {llms
      .filter((llm) => llm.id !== null && llm.id !== undefined) // Filter out invalid LLMs
      .map((llm) => (
        <SelectItem key={llm.id} value={llm.id.toString()}>
          {llm.name}
        </SelectItem>
      ))}
  </SelectContent>
</Select>
                  <Textarea
                    name="configuration"
                    placeholder='{"temperature": 0.7, "top_p": 0.9}'
                    value={form.configuration}
                    onChange={handleInputChange}
                  />
                </CardContent>
                <CardFooter>
                  <Button
                    className="w-full"
                    onClick={handleSubmit}
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Save className="mr-2 h-4 w-4" />
                    )}
                    {form.id ? "Update Agent" : "Save Agent"}
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}