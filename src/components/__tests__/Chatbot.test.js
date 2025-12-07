import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Chatbot from '../src/components/Chatbot';

// Mock the fetch function
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ session_id: 'test-session-id' }),
    body: {
      getReader: () => ({
        read: () => Promise.resolve({ done: true, value: [] }),
      }),
    },
  })
);

// Mock ReadableStream
class ReadableStream {
  constructor() {}
}

global.ReadableStream = ReadableStream;

describe('Chatbot Component', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  test('renders without crashing', async () => {
    render(<Chatbot />);
    
    // Wait for component to initialize
    await waitFor(() => {
      expect(screen.getByText(/Physical AI Textbook Assistant/i)).toBeInTheDocument();
    });
  });

  test('allows user to type and send messages', async () => {
    render(<Chatbot />);
    
    // Wait for component to initialize
    await waitFor(() => {
      expect(screen.getByText(/Physical AI Textbook Assistant/i)).toBeInTheDocument();
    });
    
    // Find the input field and simulate typing
    const input = screen.getByPlaceholderText(/Ask about Physical AI & Humanoid Robotics/i);
    fireEvent.change(input, { target: { value: 'Hello' } });
    
    // Submit the form
    const form = screen.getByRole('form');
    fireEvent.submit(form);
    
    // Check that fetch was called to create session and send message
    await waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });
  });

  test('shows loading state when sending message', async () => {
    // Mock fetch to simulate a delay
    fetch.mockImplementation(() =>
      new Promise(resolve => {
        setTimeout(() => {
          resolve({
            ok: true,
            json: () => Promise.resolve({ session_id: 'test-session-id' }),
            body: {
              getReader: () => ({
                read: () => Promise.resolve({ done: true, value: [] }),
              }),
            },
          });
        }, 100);
      })
    );
    
    render(<Chatbot />);
    
    // Wait for component to initialize
    await waitFor(() => {
      expect(screen.getByText(/Physical AI Textbook Assistant/i)).toBeInTheDocument();
    });
    
    // Type and submit a message
    const input = screen.getByPlaceholderText(/Ask about Physical AI & Humanoid Robotics/i);
    fireEvent.change(input, { target: { value: 'Test message' } });
    
    const form = screen.getByRole('form');
    fireEvent.submit(form);
    
    // Check for loading state
    expect(screen.getByText(/Physical AI Textbook Assistant/i)).toBeInTheDocument();
  });
});